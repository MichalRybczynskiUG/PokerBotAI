from evaluate_metrics import (load_model, evaluate_vs_random,
evaluate_model_vs_model, inspect_model)
from src.poker_enviroment.poker_env import PokerEnv

STATE_DIM = 302 #375
NUM_ACTIONS = 3

#model_1 = load_model("models/model_75000.pt", STATE_DIM, NUM_ACTIONS)
#model_2 = load_model("models/model_100000.pt", STATE_DIM, NUM_ACTIONS)

#evaluate_vs_random(model_2, episodes=10000)

#evaluate_model_vs_model(model_1, model_2, episodes=10000)

model = load_model(
    "models/model_100000.pt",
    STATE_DIM,
    NUM_ACTIONS
)

env = PokerEnv()
env.reset()

tests = [

    # =========
    # MONSTRY
    # =========

    (
        "Top Set",
        ["As", "Ah"],
        ["Ac", "Kd", "2c"]
    ),

    (
        "Quads",
        ["As", "Ah"],
        ["Ac", "Ad", "2c"]
    ),

    (
        "Nut Flush",
        ["As", "Qs"],
        ["2s", "7s", "Ks"]
    ),

    (
        "Nut Straight",
        ["As", "Kd"],
        ["Qh", "Jc", "Td"]
    ),

    # =========
    # MOCNE
    # =========

    (
        "Overpair",
        ["As", "Ah"],
        ["Kd", "7c", "2s"]
    ),

    (
        "Top Pair Top Kicker",
        ["As", "Ks"],
        ["Ah", "Td", "2c"]
    ),

    (
        "Two Pair",
        ["As", "Kd"],
        ["Ac", "Kh", "2s"]
    ),

    # =========
    # DRAWY
    # =========

    (
        "Nut Flush Draw",
        ["As", "Qs"],
        ["2s", "7s", "Kd"]
    ),

    (
        "Open Ended",
        ["9s", "8d"],
        ["6c", "7h", "Ks"]
    ),

    (
        "Combo Draw",
        ["As", "Qs"],
        ["Js", "Ts", "2d"]
    ),

    # =========
    # ŚREDNIE
    # =========

    (
        "Middle Pair",
        ["9s", "9d"],
        ["Ks", "9h", "Ac"]
    ),

    (
        "Weak Top Pair",
        ["8s", "7d"],
        ["8h", "Kd", "Qc"]
    ),

    # =========
    # ŚMIECI
    # =========

    (
        "Air AKQ",
        ["7c", "2d"],
        ["As", "Kd", "Qc"]
    ),

    (
        "Air Rainbow",
        ["4c", "2d"],
        ["Ks", "Qh", "9c"]
    ),

    (
        "Bottom Pair",
        ["7c", "2d"],
        ["As", "Kd", "2c"]
    ),
]

for name, hand, board in tests:

    print("\n")
    print("#" * 80)
    print(name)
    print("#" * 80)

    inspect_model(
        model,
        env,
        hand,
        board,
        model_type="dqn"
    )
"""
for name, hand, board in tests:

    print("\n")
    print("#" * 80)
    print(name)
    print("#" * 80)

    inspect_model(
        model,
        env,
        hand,
        board,
        model_type="policy"
    )
"""
extra_tests = [

    # =====================================
    # TEN SAM BOARD, RÓŻNE RĘCE
    # =====================================

    ("AKQ board - Air",
     ["3c", "2d"],
     ["As", "8h", "Qs"]),

    ("AKQ board - Top Pair",
     ["Ac", "2d"],
     ["As", "Kd", "Qc"]),

    ("AKQ board - Two Pair",
     ["Ad", "Kh"],
     ["As", "Kd", "Qc"]),

    ("AKQ board - Broadway",
     ["Js", "Td"],
     ["As", "Kd", "Qc"]),

    # =====================================
    # MONOTONE BOARD
    # =====================================

    ("Monotone Air",
     ["Ah", "Kd"],
     ["2s", "7s", "Qs"]),

    ("Monotone Flush",
     ["As", "Js"],
     ["2s", "7s", "Qs"]),

    # =====================================
    # PAIRED BOARD
    # =====================================

    ("Paired Air",
     ["7c", "2d"],
     ["Ks", "Kh", "3c"]),

    ("Paired Trips",
     ["Kd", "Qc"],
     ["Ks", "Kh", "3c"]),

    ("Paired Full House",
     ["3d", "3h"],
     ["Ks", "Kh", "3c"]),

    # =====================================
    # LOW CONNECTED BOARD
    # =====================================

    ("765 Air",
     ["As", "Kd"],
     ["7c", "6d", "5h"]),

    ("765 OESD",
     ["8s", "4d"],
     ["7c", "6d", "5h"]),

    ("765 Straight",
     ["9s", "8d"],
     ["7c", "6d", "5h"]),

    # =====================================
    # DRY ACE HIGH
    # =====================================

    ("A72 Air",
     ["7c", "2d"],
     ["As", "7h", "2c"]),

    ("A72 Top Pair",
     ["Kd", "Qs"],
     ["As", "7h", "2c"]),

    ("A72 Two Pair",
     ["7d", "2s"],
     ["As", "7h", "2c"]),

    ("A72 Set",
     ["Ac", "Ad"],
     ["As", "7h", "2c"]),
]


for name, hand, board in extra_tests:

    print("\n")
    print("#" * 80)
    print(name)
    print("#" * 80)

    inspect_model(
        model,
        env,
        hand,
        board,
        "dqn"
    )

    inspect_model(
        model,
        env,
        hand,
        board,
        "policy"
    )
