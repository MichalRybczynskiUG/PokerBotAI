import torch
import numpy as np
from action_mapper import ACTION_MAP, map_to_env
from src.bot.random_bot import RandomBot
from bot_architecture import (
    NFSPModel,
    select_action_nfsp,
    compute_q_loss,
    compute_policy_loss,
    build_optimizers
)
from memory import ReplayBuffer
from src.poker_enviroment.poker_env import PokerEnv
from src.poker_enviroment.observation import legal_action_mask


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
        self.env = PokerEnv()
        self.random_bot = RandomBot()

        self.model = NFSPModel(state_dim, num_actions)
        self.q_opt, self.policy_opt = build_optimizers(self.model)

        self.rl_buffer = ReplayBuffer(10000)
        self.sl_buffer = ReplayBuffer(10000)

        self.gamma = 0.99
        self.update_target_every = 1000
        self.step_count = 0

    def play_episode(self, eta=0.1):
        """Run a single episode and collect experiences.

        Plays one full game in the environment, selecting actions using NFSP.
        Depending on the mode (best response or average policy), transitions
        are later stored in appropriate buffers for training.

        Args:
            eta (float, optional): Probability of using best response (Q-network).
                Defaults to 0.1.

        Returns:
            None
        """
        state = self.env.reset()
        done = False

        while not done:

            player = self.env.current_player

            legal_mask = legal_action_mask(self.env.legal_actions())

            action, mode = select_action_nfsp(
                self.model,
                state,
                legal_mask,
                eta=eta
            )

            symbol = ACTION_MAP[action]
            env_action, raise_amount = map_to_env(symbol, self.env)

            if raise_amount is None and env_action == 2:
                raise_amount = max(2, int(self.env.engine.to_call))

            next_state, reward, done, _ = self.env.step(env_action, raise_amount)

            next_state_fixed = self.env._get_observation(player)

            self.rl_buffer.push(state, action, reward, next_state_fixed, done)

            if mode == "BR":
                self.sl_buffer.push(state, action)

            state = next_state
            self.step_count += 1

            if self.step_count % self.update_target_every == 0:
                self.model.update_target()

    def train_q(self, batch_size=64):
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
        if len(self.rl_buffer) < batch_size:
            return None

        batch = self.rl_buffer.sample(batch_size)
        loss = compute_q_loss(self.model, batch, self.gamma)

        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()

        return loss.item()

    def train_policy(self, batch_size=64):
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
        if len(self.sl_buffer) < batch_size:
            return None

        batch = self.sl_buffer.sample(batch_size)
        loss = compute_policy_loss(self.model, batch)

        self.policy_opt.zero_grad()
        loss.backward()
        self.policy_opt.step()

        return loss.item()

    def train(self, episodes=10000, eval_every=300, eval_episodes=100):
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

            #self-play
            self.play_episode()

            #update models
            loss_q = self.train_q()
            loss_policy = self.train_policy()

            #logs
            if ep % 100 == 0:
                print(f"\nEpisode {ep}")

                if loss_q is not None:
                    print(f"Q loss: {loss_q:.6f}")

                if loss_policy is not None:
                    print(f"Policy loss: {loss_policy:.6f}")

                print(f"RL buffer: {len(self.rl_buffer)} | SL buffer: {len(self.sl_buffer)}")

            # evaluate on random bot
            if ep % eval_every == 0 and ep > 0:
                winrate = self.evaluate_vs_random(episodes=eval_episodes)

                print(f"Eval vs random: {winrate:.4f} ({eval_episodes} games)")

                torch.save(self.model.state_dict(), f"model_{ep}.pt")

    def evaluate_vs_random(self, episodes=100):
        """Evaluate the NFSP agent against a random baseline.

        Plays multiple episodes where the NFSP model competes against a
        RandomBot. The starting position alternates between players to
        ensure fairness. The win rate is computed based on final stack sizes.

        Args:
            episodes (int, optional): Number of evaluation games.
                Defaults to 100.

        Returns:
            float: Win rate of the NFSP agent (0.0–1.0).
        """
        wins = 0

        for ep in range(episodes):
            state = self.env.reset()
            done = False

            model_is_p1 = (ep % 2 == 0)

            while not done:
                if (model_is_p1 and self.env.current_player == self.env.p1) or \
                        (not model_is_p1 and self.env.current_player == self.env.p2):

                    legal_mask = legal_action_mask(self.env.legal_actions())

                    action, _ = select_action_nfsp(
                        self.model,
                        state,
                        legal_mask,
                        eta=0.0,
                        epsilon=0.02
                    )

                    symbol = ACTION_MAP[action]
                    env_action, raise_amount = map_to_env(symbol, self.env)

                    if raise_amount is None and env_action == 2:
                        raise_amount = max(2, int(self.env.engine.to_call))

                else:
                    result = self.random_bot.select_action(self.env)

                    if isinstance(result, tuple):
                        env_action, raise_amount = result
                    else:
                        symbol = ACTION_MAP[result]
                        env_action, raise_amount = map_to_env(symbol, self.env)

                state, reward, done, _ = self.env.step(env_action, raise_amount)
                print(self.env.p1.stack, self.env.p2.stack)

            if model_is_p1:
                if self.env.p1.stack > self.env.p2.stack:
                    wins += 1
            else:
                if self.env.p2.stack > self.env.p1.stack:
                    wins += 1
        print(f"Wins: {wins}")

        return wins / episodes