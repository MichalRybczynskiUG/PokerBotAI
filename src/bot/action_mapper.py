from src.poker_enviroment.constants import (
    ACTION_FOLD,
    ACTION_CALL,
    ACTION_BET_25,
    ACTION_BET_33,
    ACTION_BET_50,
    ACTION_BET_75,
    ACTION_BET_100,
    ACTION_ALL_IN,
)

def map_to_env(action, env):
    pot = env.engine.pot

    if action == ACTION_FOLD:
        return ACTION_FOLD, None

    elif action == ACTION_CALL:
        return ACTION_CALL, None

    elif action == ACTION_BET_25:
        return action, max(1, int(pot * 0.25))

    elif action == ACTION_BET_33:
        return action, max(1, int(pot * 0.33))

    elif action == ACTION_BET_50:
        return action, max(1, int(pot * 0.5))

    elif action == ACTION_BET_75:
        return action, max(1, int(pot * 0.75))

    elif action == ACTION_BET_100:
        return action, max(1, int(pot))

    elif action == ACTION_ALL_IN:
        return action, None
    else:
        raise ValueError(f"Unknown action: {action}")