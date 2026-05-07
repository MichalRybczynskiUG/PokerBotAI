import torch
from action_mapper import map_to_env
from src.bot.random_bot import RandomBot
from bot_architecture import (
    NFSPModel,
    select_action_nfsp,
    compute_q_loss,
    compute_policy_loss,
    build_optimizers
)
from memory import ReplayBuffer,ReservoirBuffer
from src.poker_enviroment.poker_env import PokerEnv
from src.poker_enviroment.observation import legal_action_mask
from evaluate_metrics import evaluate_vs_random


class NFSPTrainer:
    """Trainer for NFSP agent in a poker environment.

    Handles environment interaction, experience collection, and training
    of both Q-network (reinforcement learning) and policy network
    (supervised learning).
    """

    def __init__(self, state_dim, num_actions):
        """Initialize trainer components.

        Args:
            state_dim (int): Dimensionality of the input state.
            num_actions (int): Number of discrete actions.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.env = PokerEnv()
        self.random_bot = RandomBot()

        self.model_p1 = NFSPModel(state_dim, num_actions).to(self.device)
        self.model_p2 = NFSPModel(state_dim, num_actions).to(self.device)

        self.q_opt_p1, self.policy_opt_p1 = build_optimizers(self.model_p1)
        self.q_opt_p2, self.policy_opt_p2 = build_optimizers(self.model_p2)

        self.rl_buffer_p1 = ReplayBuffer(600_000)
        self.sl_buffer_p1 = ReservoirBuffer(1_000_000)

        self.rl_buffer_p2 = ReplayBuffer(600_000)
        self.sl_buffer_p2 = ReservoirBuffer(1_000_000)

        self.gamma = 0.99
        self.update_target_every = 1000
        self.step_count = 0

    def play_episode(self, eta=0.1):
        state = self.env.reset().to(self.device)
        done = False

        while not done:

            current = self.env.current_player

            if current.all_in:
                state, reward, done, _ = self.env.step(1, None)
                state = state.to(self.device)
                continue

            epsilon = max(0.01, 0.15 * (1 - self.step_count / 5_000_000))

            legal = self.env.legal_actions()

            legal_mask = torch.tensor(
                legal_action_mask(legal),
                dtype=torch.float32,
                device=self.device
            )

            if current == self.env.p1:
                model = self.model_p1
                rl_buffer = self.rl_buffer_p1
                sl_buffer = self.sl_buffer_p1
            else:
                model = self.model_p2
                rl_buffer = self.rl_buffer_p2
                sl_buffer = self.sl_buffer_p2

            action, mode = select_action_nfsp(
                model,
                state,
                legal_mask,
                eta=eta,
                epsilon=epsilon
            )

            env_action, raise_amount = map_to_env(action, self.env)

            next_state, reward, done, _ = self.env.step(env_action, raise_amount)
            next_state = next_state.to(self.device)

            if not done:
                next_legal_mask = torch.tensor(
                    legal_action_mask(self.env.legal_actions()),
                    dtype=torch.float32,
                    device=self.device
                )
            else:
                next_legal_mask = torch.ones_like(legal_mask, device=self.device)

            rl_buffer.push(
                state,
                action,
                reward,
                next_state,
                done,
                next_legal_mask
            )

            if mode == "BR":
                sl_buffer.push(state, action, legal_mask)

            state = next_state
            self.step_count += 1

            if self.step_count % 256 == 0:

                for _ in range(2):
                    self.train_q(self.model_p1, self.rl_buffer_p1, self.q_opt_p1)
                    self.train_q(self.model_p2, self.rl_buffer_p2, self.q_opt_p2)
                    self.train_policy(self.model_p1, self.sl_buffer_p1, self.policy_opt_p1)
                    self.train_policy(self.model_p2, self.sl_buffer_p2, self.policy_opt_p2)

            if self.step_count % 1000 == 0:
                self.model_p1.update_target()
                self.model_p2.update_target()

    def train_q(self, model, buffer, optimizer, batch_size=256):
        """Train the Q-network using a batch from the replay buffer.

        Samples transitions from the RL buffer and performs a single
        gradient update step using the Q-learning loss.

        Args:
            batch_size (int, optional): Number of samples per batch.
                Defaults to 64.

        Returns:
            float or None: Loss value if training was performed,
            otherwise None if not enough data.
        """
        if len(buffer) < batch_size:
            return None

        batch = buffer.sample(batch_size)

        states, actions, rewards, next_states, dones, next_masks = batch

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        next_masks = torch.stack(next_masks).to(self.device)

        batch = (states, actions, rewards, next_states, dones, next_masks)

        loss = compute_q_loss(model, batch, self.gamma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def train_policy(self, model, buffer, optimizer, batch_size=256):
        """Train the policy network using supervised learning buffer.

        Samples state-action pairs from the SL buffer and performs a
        gradient update to improve the average policy.

        Args:
            batch_size (int, optional): Number of samples per batch.
                Defaults to 64.

        Returns:
            float or None: Loss value if training was performed,
            otherwise None if not enough data.
        """
        if len(buffer) < batch_size:
            return None

        batch = buffer.sample(batch_size)

        states, actions, masks = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        masks = torch.stack(masks).to(self.device)

        loss = compute_policy_loss(model, (states, actions), masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self, episodes=250100, eval_every=50000, eval_episodes=5000):
        """Train the NFSP agent through self-play.

        Runs multiple episodes of self-play, updates both Q-network and
        policy network, logs training progress, and periodically evaluates
        performance against a random baseline agent.

        Args:
            episodes (int, optional): Total number of training episodes.
                Defaults to 10000.
            eval_every (int, optional): Evaluation frequency (in episodes).
                Defaults to 300.
            eval_episodes (int, optional): Number of games per evaluation.
                Defaults to 100.

        """
        for ep in range(episodes):

            self.play_episode()

            if ep % eval_episodes == 0:
                print(f"\nEpisode {ep}")
                print(
                    f"RL: P1={len(self.rl_buffer_p1)} P2={len(self.rl_buffer_p2)} | "
                    f"SL: P1={len(self.sl_buffer_p1)} P2={len(self.sl_buffer_p2)}"
                )

            if ep % eval_every == 0 and ep > 0:
                self.save_all(prefix=f"checkpoints/checkpoint_{ep}")
                self.save_models_only(prefix=f"models/model_{ep}")

                print(f"\n=== EVAL at episode {ep} ===")

                ev, bb100, mbb = evaluate_vs_random(self.model_p1, episodes=1000)

                print(f"EV: {ev:.2f} | bb/100: {bb100:.2f} | mbb/h: {mbb:.2f}")

    def save_all(self, prefix="checkpoint"):
        checkpoint = {
            "model_p1": self.model_p1.state_dict(),
            "model_p2": self.model_p2.state_dict(),

            "q_opt_p1": self.q_opt_p1.state_dict(),
            "q_opt_p2": self.q_opt_p2.state_dict(),
            "policy_opt_p1": self.policy_opt_p1.state_dict(),
            "policy_opt_p2": self.policy_opt_p2.state_dict(),

            "step_count": self.step_count,

            "rl_buffer_p1": self.rl_buffer_p1.buffer,
            "rl_buffer_p2": self.rl_buffer_p2.buffer,
            "sl_buffer_p1": self.sl_buffer_p1.buffer,
            "sl_buffer_p2": self.sl_buffer_p2.buffer,
        }

        torch.save(checkpoint, f"{prefix}.pt")
        print("Saved SINGLE FILE checkpoint")

    def load_for_train(self, prefix="checkpoint"):
        checkpoint = torch.load(f"{prefix}.pt", map_location=self.device)

        # modele
        self.model_p1.load_state_dict(checkpoint["model_p1"])
        self.model_p2.load_state_dict(checkpoint["model_p2"])

        # optymalizatory
        self.q_opt_p1.load_state_dict(checkpoint["q_opt_p1"])
        self.q_opt_p2.load_state_dict(checkpoint["q_opt_p2"])
        self.policy_opt_p1.load_state_dict(checkpoint["policy_opt_p1"])
        self.policy_opt_p2.load_state_dict(checkpoint["policy_opt_p2"])

        self.step_count = checkpoint["step_count"]

        self.rl_buffer_p1.buffer = checkpoint["rl_buffer_p1"]
        self.rl_buffer_p2.buffer = checkpoint["rl_buffer_p2"]
        self.sl_buffer_p1.buffer = checkpoint["sl_buffer_p1"]
        self.sl_buffer_p2.buffer = checkpoint["sl_buffer_p2"]

        print("Loaded SINGLE FILE checkpoint")

    def save_models_only(self, prefix="model"):
        checkpoint = {
            "model_p1": self.model_p1.state_dict(),
            "model_p2": self.model_p2.state_dict(),
        }

        torch.save(checkpoint, f"{prefix}.pt")
        print("Saved LIGHT checkpoint (models only)")