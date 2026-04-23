from src.game_abstraction.betowanie_abstrakcja import (
    CHECK, BET_MIN, BET_MAX, CALL, FOLD
)

ACTION_MAP = {
    0: FOLD,
    1: CALL,
    2: BET_MIN,
    3: BET_MAX,
}

def map_to_env(action_symbol, env):
    player = env.current_player
    pot = env.engine.pot

    if action_symbol == FOLD:
        return 0, None

    elif action_symbol == CALL:
        return 1, None

    elif action_symbol == BET_MIN:
        raise_amount = max(10, int(pot * 0.33))
        return 2, raise_amount

    elif action_symbol == BET_MAX:
        raise_amount = max(10, int(pot))
        return 2, raise_amount

    else:
        raise ValueError("Unknown action")