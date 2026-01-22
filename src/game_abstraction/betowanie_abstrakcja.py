"""
Standalone Betting Abstraction for No-Limit Poker
(no typing imports)

Defines:
- abstract actions
- allowed betting sequences
- transition rules (DFA)
- terminal detection
- mapping from real bet sizes to abstract actions
"""

# --------------------------------------------------
# 1. Abstract Action Space
# --------------------------------------------------

CHECK = "k"
BET_MIN = "bMIN"   # ~1/3 pot (or minimum bet)
BET_MAX = "bMAX"   # pot-sized bet
CALL = "c"
FOLD = "f"

ACTIONS = [CHECK, BET_MIN, BET_MAX, CALL, FOLD]

# --------------------------------------------------
# 2. Terminal Betting Sequences (11 total)
# --------------------------------------------------

TERMINAL_SEQUENCES = {
    (CHECK, CHECK),

    (CHECK, BET_MIN, CALL),
    (CHECK, BET_MIN, FOLD),

    (CHECK, BET_MAX, CALL),
    (CHECK, BET_MAX, FOLD),

    (BET_MIN, CALL),
    (BET_MIN, FOLD),

    (BET_MIN, BET_MAX, CALL),
    (BET_MIN, BET_MAX, FOLD),

    (BET_MAX, CALL),
    (BET_MAX, FOLD),
}

# --------------------------------------------------
# 3. Transition Function (Deterministic FSM)
# --------------------------------------------------

TRANSITIONS = {
    (): [CHECK, BET_MIN, BET_MAX],

    (CHECK,): [CHECK, BET_MIN, BET_MAX],
    (CHECK, CHECK): [],

    (CHECK, BET_MIN): [CALL, FOLD],
    (CHECK, BET_MAX): [CALL, FOLD],

    (BET_MIN,): [CALL, FOLD, BET_MAX],
    (BET_MIN, BET_MAX): [CALL, FOLD],

    (BET_MAX,): [CALL, FOLD],
}

# --------------------------------------------------
# 4. Betting Abstraction Core
# --------------------------------------------------

class BettingAbstraction:
    """
    Deterministic betting abstraction.
    History = list of abstract action symbols.
    """

    def legal_actions(self, history):
        return TRANSITIONS.get(tuple(history), [])

    def is_terminal(self, history):
        return tuple(history) in TERMINAL_SEQUENCES

    def is_valid(self, history):
        prefix = []
        for action in history:
            if action not in self.legal_actions(prefix):
                return False
            prefix.append(action)
        return True

    def reset(self):
        return []

    def all_terminal_sequences(self):
        return [list(seq) for seq in TERMINAL_SEQUENCES]

# --------------------------------------------------
# 5. Real Bet Size → Abstract Action Mapping
# --------------------------------------------------

class BetSizeMapper:
    """
    Lossy compression from real bet sizes to abstract actions.
    """

    def __init__(self, min_fraction=1/3):
        self.min_fraction = min_fraction

    def map_bet(self, bet_amount, pot_size):
        if pot_size <= 0:
            return BET_MIN

        threshold = max(2.0, pot_size * self.min_fraction)
        if bet_amount <= threshold:
            return BET_MIN
        else:
            return BET_MAX

    def map_sequence(self, bet_sequence, pot_sequence):
        abstraction = BettingAbstraction()
        history = []

        for bet, pot in zip(bet_sequence, pot_sequence):
            action = self.map_bet(bet, pot)
            legal = abstraction.legal_actions(history)

            if action not in legal:
                if BET_MAX in legal:
                    action = BET_MAX
                else:
                    break

            history.append(action)

            if abstraction.is_terminal(history):
                break

        return history

