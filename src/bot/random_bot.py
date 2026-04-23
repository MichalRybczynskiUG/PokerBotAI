import random

class RandomBot:
    """Simple baseline agent that selects random legal actions.

    If a raise action is chosen, a random valid raise amount is sampled.
    """

    def select_action(self, env):
        """Select a random legal action.

        Args:
            env: Environment providing legal actions and game state.

        Returns:
            tuple:
                int: Chosen action.
                int or None: Raise amount if action is a raise, otherwise None.
        """
        legal = env.legal_actions()
        action = random.choice(legal)

        if action == 2:
            min_raise = max(2, env.engine.to_call)
            max_raise = env.current_player.stack

            if min_raise >= max_raise:
                return action, min_raise

            amount = random.randint(min_raise, max_raise)
            return action, amount

        return action, None