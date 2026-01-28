SUITS = ["♠", "♥", "♦", "♣"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SMALL_BLIND = 5
BIG_BLIND = 10
MIN_RAISE = BIG_BLIND

ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_RAISE = 2
ACTION_ALL_IN = 3

ACTION_NAMES = {
    0: "fold",
    1: "call/check",
    2: "raise",
    3: "all-in"
}

PREFLOP = 0
FLOP = 1
TURN = 2
RIVER = 3
SHOWDOWN = 4

STREET_NAMES = {
    0 : "PREFLOP",
    1 : "FLOP",
    2 : "TURN",
    3 : "RIVER",
    4 : "SHOWDOWN",
}