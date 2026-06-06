import random
from src.poker_enviroment.constants import (
    ACTION_FOLD,
    ACTION_CALL,
    ACTION_BET_50,
    ACTION_BET_100,
    ACTION_ALL_IN,
)

class RandomBot:
    """Simple baseline agent that selects random legal actions.

    If a raise action is chosen, a random valid raise amount is sampled.
    """

    def select_action(self, env):
        legal = env.legal_actions()
        action = random.choice(legal)

        pot = env.engine.pot

        # FOLD / CALL
        if action in [ACTION_FOLD, ACTION_CALL]:
            return action, None

        elif action == ACTION_BET_50:
            return action, max(1, int(pot * 0.5))

        elif action == ACTION_BET_100:
            return action, max(1, int(pot))

        # ALL-IN
        elif action == ACTION_ALL_IN:
            return action, None

        # fallback (nie powinno się zdarzyć)
        return ACTION_CALL, None