import torch
from bot_architecture import NFSPModel, select_action_nfsp
from src.poker_enviroment.poker_env import PokerEnv
from src.poker_enviroment.observation import legal_action_mask
from action_mapper import map_to_env
import numpy as np
from src.bot.random_bot import RandomBot


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
    action_counts = np.zeros(8)

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

            # skip ALL-IN
            if current.all_in:
                _, _, done, _ = env.step(1, None)
                state = get_model_state()
                continue

            legal = env.legal_actions()

            legal_mask = torch.tensor(
                legal_action_mask(legal),
                dtype=torch.float32,
                device=device
            )

            if (model_is_p1 and current == env.p1) or \
                    (not model_is_p1 and current == env.p2):

                legal = env.legal_actions()

                legal_mask = torch.tensor(
                    legal_action_mask(legal),
                    dtype=torch.float32,
                    device=device
                )

                action, _ = select_action_nfsp(
                    model,
                    state,
                    legal_mask,
                    eta=0.0,
                    epsilon=0.0
                )

                action_counts[action] += 1

                env_action, raise_amount = map_to_env(action, env)

            else:
                result = random_bot.select_action(env)

                if isinstance(result, tuple):
                    env_action, raise_amount = result
                else:
                    env_action, raise_amount = map_to_env(result, env)

            _, _, done, _ = env.step(env_action, raise_amount)

            state = get_model_state()

        if model_is_p1:
            profit = env.p1.stack - start_stack_p1
        else:
            profit = env.p2.stack - start_stack_p2

        total_profit += profit

    ev = total_profit / episodes

    if action_counts.sum() > 0:
        print("Action distribution:", action_counts / action_counts.sum())

    print(f"EV: {ev:.2f}")

    return ev

def evaluate_model_vs_model(model_A, model_B, episodes=5000):
    env = PokerEnv()
    total_profit = 0
    wins_A = 0

    action_counts_A = torch.zeros(8)
    action_counts_B = torch.zeros(8)

    for ep in range(episodes):
        state = env.reset().to(device)
        done = False

        A_is_p1 = (ep % 2 == 0)

        while not done:
            current = env.current_player

            if current.all_in:
                state, _, done, _ = env.step(1, None)
                state = state.to(device)
                continue

            legal = env.legal_actions()

            legal_mask = torch.tensor(
                legal_action_mask(legal),
                dtype=torch.float32,
                device=device
            )

            if (A_is_p1 and current == env.p1) or \
               (not A_is_p1 and current == env.p2):
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

            env_action, raise_amount = map_to_env(action, env)
            state, _, done, _ = env.step(env_action, raise_amount)
            state = state.to(device)

        if A_is_p1:
            profit = env.p1.stack - env.initial_stack
        else:
            profit = env.p2.stack - env.initial_stack

        total_profit += profit
        if profit > 0:
            wins_A += 1

        if (ep + 1) % 1000 == 0:
            print(f"[{ep+1}/{episodes}] EV: {total_profit/(ep+1):.4f}")

    ev = total_profit / episodes
    winrate = wins_A / episodes

    probs_A = action_counts_A / (action_counts_A.sum() + 1e-8)
    probs_B = action_counts_B / (action_counts_B.sum() + 1e-8)

    print("\n=== FINAL ===")
    print(f"EV: {ev:.4f}")
    print(f"Winrate A: {winrate:.3f}")

    print("\n=== ACTION DISTRIBUTION ===")
    print("Model A:", probs_A.cpu().numpy())
    print("Model B:", probs_B.cpu().numpy())

    entropy_A = -(probs_A * torch.log(probs_A + 1e-8)).sum()
    entropy_B = -(probs_B * torch.log(probs_B + 1e-8)).sum()

    print(f"\nEntropy A: {entropy_A.item():.4f}")
    print(f"Entropy B: {entropy_B.item():.4f}")

    return ev