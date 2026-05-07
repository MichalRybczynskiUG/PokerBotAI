SUITS = ["s", "h", "d", "c"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SMALL_BLIND = 5
BIG_BLIND = 10
MIN_RAISE = BIG_BLIND

ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_BET_25 = 2
ACTION_BET_33 = 3
ACTION_BET_50 = 4
ACTION_BET_75 = 5
ACTION_BET_100 = 6
ACTION_ALL_IN = 7

NUM_ACTIONS = 8

ACTION_NAMES = {
    0: "fold",
    1: "call/check",
    2: "bet_25",
    3: "bet_33",
    4: "bet_50",
    5: "bet_75",
    6: "bet_100",
    7: "all-in"
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