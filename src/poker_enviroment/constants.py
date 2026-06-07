SUITS = ["s", "h", "d", "c"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

SMALL_BLIND = 5
BIG_BLIND = 10

SMALL_BET = 10
BIG_BET = 20

MAX_RAISES = 4

ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_RAISE = 2

NUM_ACTIONS = 3

ACTION_NAMES = {
    0: "fold",
    1: "call/check",
    2: "raise"
}

PREFLOP = 0
FLOP = 1
TURN = 2
RIVER = 3
SHOWDOWN = 4