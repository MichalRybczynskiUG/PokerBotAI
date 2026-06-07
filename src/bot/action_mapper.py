from src.poker_enviroment.constants import (
    ACTION_FOLD,
    ACTION_CALL,
    ACTION_RAISE,
)

def map_to_env(action, env):

    if action == ACTION_FOLD:
        return ACTION_FOLD

    elif action == ACTION_CALL:
        return ACTION_CALL

    elif action == ACTION_RAISE:
        return ACTION_RAISE

    raise ValueError(f"Unknown action: {action}")