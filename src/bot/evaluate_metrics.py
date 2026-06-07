import torch
from bot_architecture import NFSPModel, select_action_nfsp
from src.poker_enviroment.poker_env import PokerEnv
from action_mapper import map_to_env
from src.bot.random_bot import RandomBot

from collections import defaultdict
import numpy as np
from src.poker_enviroment.observation import legal_action_mask, hand_to_ids
from src.poker_enviroment.constants import PREFLOP
from src.poker_enviroment.constants import ACTION_CALL


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path, state_dim, num_actions):
    model = NFSPModel(state_dim, num_actions).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_p1"])
    return model

def evaluate_vs_random(model, episodes=1000):

    env = PokerEnv()
    random_bot = RandomBot()

    device = next(model.parameters()).device

    total_profit = 0.0

    action_counts = np.zeros(3)
    q_sums = np.zeros(3)

    q_count = 0

    for ep in range(episodes):

        env.reset()

        start_stack_p1 = env.initial_stack_p1
        start_stack_p2 = env.initial_stack_p2

        done = False

        model_is_p1 = (ep % 2 == 0)

        def get_model_state():
            player = env.p1 if model_is_p1 else env.p2
            return env._get_observation(player).to(device)

        state = get_model_state()

        while not done:

            current = env.current_player

            if current.all_in:
                _, _, done, _ = env.step(ACTION_CALL)
                state = get_model_state()
                continue

            legal = env.legal_actions()

            legal_mask = torch.tensor(
                legal_action_mask(legal),
                dtype=torch.float32,
                device=device
            )

            if (
                (model_is_p1 and current == env.p1)
                or
                (not model_is_p1 and current == env.p2)
            ):

                with torch.no_grad():
                    q = model.q_net(
                        state.unsqueeze(0)
                    ).squeeze(0)

                    q_sums += q.cpu().numpy()
                    q_count += 1

                action, _ = select_action_nfsp(
                    model,
                    state,
                    legal_mask,
                    eta=0.0,
                    epsilon=0.0
                )

                action_counts[action] += 1

                env_action = map_to_env(
                    action,
                    env
                )

            else:

                env_action = random_bot.select_action(
                    env
                )

            _, _, done, _ = env.step(
                env_action
            )

            state = get_model_state()

        if model_is_p1:
            profit = env.p1.stack - start_stack_p1
        else:
            profit = env.p2.stack - start_stack_p2

        total_profit += profit

    ev = total_profit / episodes

    if action_counts.sum() > 0:
        print(
            "Action distribution:",
            action_counts / action_counts.sum()
        )

    if q_count > 0:

        avg_q = q_sums / q_count

        print("\nAverage Q values:")

        print(
            f"fold={avg_q[0]:.3f} "
            f"call={avg_q[1]:.3f} "
            f"raise={avg_q[2]:.3f}"
        )

    print(f"EV: {ev:.2f}")

    return ev


def evaluate_model_vs_model(model_A, model_B, episodes=5000):

    env = PokerEnv()

    total_profit = 0
    wins_A = 0

    action_counts_A = torch.zeros(3)
    action_counts_B = torch.zeros(3)

    for ep in range(episodes):

        state = env.reset().to(device)

        done = False

        A_is_p1 = (ep % 2 == 0)

        while not done:

            current = env.current_player

            if current.all_in:

                state, _, done, _ = env.step(
                    ACTION_CALL
                )

                state = state.to(device)

                continue

            legal = env.legal_actions()

            legal_mask = torch.tensor(
                legal_action_mask(legal),
                dtype=torch.float32,
                device=device
            )

            if (
                (A_is_p1 and current == env.p1)
                or
                (not A_is_p1 and current == env.p2)
            ):
                model = model_A
                is_A_turn = True
            else:
                model = model_B
                is_A_turn = False

            action, _ = select_action_nfsp(
                model,
                state,
                legal_mask,
                eta=0.0,
                epsilon=0.0
            )

            if is_A_turn:
                action_counts_A[action] += 1
            else:
                action_counts_B[action] += 1

            env_action = map_to_env(
                action,
                env
            )

            state, _, done, _ = env.step(
                env_action
            )

            state = state.to(device)

        if A_is_p1:
            profit = (
                env.p1.stack
                - env.initial_stack_p1
            )
        else:
            profit = (
                env.p2.stack
                - env.initial_stack_p2
            )

        total_profit += profit

        if profit > 0:
            wins_A += 1

        if (ep + 1) % 1000 == 0:

            print(
                f"[{ep+1}/{episodes}] "
                f"EV: {total_profit/(ep+1):.4f}"
            )

    ev = total_profit / episodes
    winrate = wins_A / episodes

    probs_A = (
        action_counts_A
        / (action_counts_A.sum() + 1e-8)
    )

    probs_B = (
        action_counts_B
        / (action_counts_B.sum() + 1e-8)
    )

    print("\n=== FINAL ===")
    print(f"EV: {ev:.4f}")
    print(f"Winrate A: {winrate:.3f}")

    print("\n=== ACTION DISTRIBUTION ===")

    print(
        "Model A:",
        probs_A.cpu().numpy()
    )

    print(
        "Model B:",
        probs_B.cpu().numpy()
    )

    entropy_A = -(
        probs_A
        * torch.log(probs_A + 1e-8)
    ).sum()

    entropy_B = -(
        probs_B
        * torch.log(probs_B + 1e-8)
    ).sum()

    print(
        f"\nEntropy A: "
        f"{entropy_A.item():.4f}"
    )

    print(
        f"Entropy B: "
        f"{entropy_B.item():.4f}"
    )

    return ev

def analyze_starting_hands(model, episodes=50000):

    env = PokerEnv()
    random_bot = RandomBot()

    device = next(model.parameters()).device

    hand_action_stats = defaultdict(
        lambda: np.zeros(3)
    )

    hand_ev_stats = defaultdict(list)

    for ep in range(episodes):

        env.reset()

        model_is_p1 = (ep % 2 == 0)

        if model_is_p1:
            tracked_hand = hand_to_ids(
                env.p1.hand[0],
                env.p1.hand[1]
            )
        else:
            tracked_hand = hand_to_ids(
                env.p2.hand[0],
                env.p2.hand[1]
            )

        start_stack_p1 = env.initial_stack_p1
        start_stack_p2 = env.initial_stack_p2

        logged_action = False
        done = False

        while not done:

            current = env.current_player

            if current.all_in:
                _, _, done, _ = env.step(
                    ACTION_CALL
                )
                continue

            legal = env.legal_actions()

            legal_mask = torch.tensor(
                legal_action_mask(legal),
                dtype=torch.float32,
                device=device
            )

            if (
                (model_is_p1 and current == env.p1)
                or
                (not model_is_p1 and current == env.p2)
            ):

                state = env._get_observation(
                    current
                ).to(device)

                action, _ = select_action_nfsp(
                    model,
                    state,
                    legal_mask,
                    eta=0.0,
                    epsilon=0.0
                )

                if (
                    not logged_action
                    and env.street == PREFLOP
                ):
                    hand_action_stats[
                        tracked_hand
                    ][action] += 1

                    logged_action = True

                env_action = map_to_env(
                    action,
                    env
                )

            else:

                env_action = random_bot.select_action(
                    env
                )

            _, _, done, _ = env.step(
                env_action
            )

        if model_is_p1:
            profit = (
                env.p1.stack
                - start_stack_p1
            )
        else:
            profit = (
                env.p2.stack
                - start_stack_p2
            )

        hand_ev_stats[
            tracked_hand
        ].append(profit)

    print("\n===== STARTING HAND ANALYSIS =====\n")

    results = []

    for hand in hand_ev_stats:

        profits = np.array(
            hand_ev_stats[hand]
        )

        n = len(profits)

        if n < 50:
            continue

        avg_ev = profits.mean()
        std_ev = profits.std()
        sem_ev = std_ev / np.sqrt(n)

        min_ev = profits.min()
        max_ev = profits.max()

        probs = (
            hand_action_stats[hand]
            / max(
                hand_action_stats[hand].sum(),
                1
            )
        )

        results.append(
            (
                avg_ev,
                std_ev,
                sem_ev,
                min_ev,
                max_ev,
                hand,
                n,
                probs
            )
        )

    results.sort(reverse=True)

    print(
        f"{'Hand':<5} "
        f"{'EV':>8} "
        f"{'STD':>8} "
        f"{'SEM':>8} "
        f"{'N':>6} "
        f"{'MIN':>8} "
        f"{'MAX':>8} "
        f"{'Fold':>7} "
        f"{'Call':>7} "
        f"{'Raise':>7}"
    )

    print("-" * 100)

    for (
        avg_ev,
        std_ev,
        sem_ev,
        min_ev,
        max_ev,
        hand,
        n,
        probs
    ) in results:

        print(
            f"{hand:<5} "
            f"{avg_ev:8.2f} "
            f"{std_ev:8.2f} "
            f"{sem_ev:8.2f} "
            f"{n:6d} "
            f"{min_ev:8.0f} "
            f"{max_ev:8.0f} "
            f"{probs[0]:7.2f} "
            f"{probs[1]:7.2f} "
            f"{probs[2]:7.2f}"
        )

    return results

def inspect_model(
    model,
    env,
    hand,
    board=None,
    model_type="dqn"
):

    if board is None:
        board = []

    device = next(
        model.parameters()
    ).device

    env.reset()

    env.p1.hand = hand
    env.board = board

    state = env._get_observation(
        env.p1
    ).to(device)

    actions = [
        "fold",
        "call",
        "raise"
    ]

    with torch.no_grad():

        if model_type == "dqn":

            values = model.q_net(
                state.unsqueeze(0)
            )[0]

            z = model.q_net.encoder(
                state.unsqueeze(0)
            )[0]

            label = "Q values"

        elif model_type == "policy":

            logits = model.policy_net(
                state.unsqueeze(0)
            )[0]

            values = torch.softmax(
                logits,
                dim=-1
            )

            z = model.policy_net.encoder(
                state.unsqueeze(0)
            )[0]

            label = "Probabilities"

        else:
            raise ValueError(
                "model_type must be "
                "'dqn' or 'policy'"
            )

    best_action = torch.argmax(
        values
    ).item()

    print()
    print("=" * 60)

    print("Hand :", hand)
    print("Board:", board)

    print(f"\n{label}:")

    for a, v in zip(actions, values):

        print(
            f"{a:>6}: "
            f"{v.item():8.3f}"
        )

    print(
        "\nBest action:",
        actions[best_action]
    )

    print(
        "\nEmbedding norm:",
        round(
            torch.norm(z).item(),
            3
        )
    )

    print("=" * 60)

    return {
        "values": values.cpu(),
        "embedding": z.cpu(),
        "best_action": actions[
            best_action
        ]
    }
    print("=" * 60)

    return {
        "values": values.cpu(),
        "embedding": z.cpu(),
        "best_action": actions[
            best_action
        ]
    }