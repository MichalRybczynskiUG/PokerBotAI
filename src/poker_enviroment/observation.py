from pathlib import Path
import pickle
import numpy as np
from src.poker_enviroment.constants import NUM_ACTIONS, RANKS, SUITS
from treys import Card
from src.game_abstraction.board_features import extract_flop_board_features
from src.game_abstraction.board_features import extract_turn_board_features
SUIT_MAP = {
    "♠": "s",
    "♥": "h",
    "♦": "d",
    "♣": "c"
}

def to_treys(card_str):
    rank = card_str[0]
    suit = SUIT_MAP[card_str[1]]
    return rank + suit


def get_flop_bucket(board, abstraction):
    kmeans = abstraction["kmeans"]
    mean = abstraction["mean"]
    std = abstraction["std"]

    c1, c2, c3 = [Card.new(to_treys(c)) for c in board]

    vec = extract_flop_board_features(c1, c2, c3)
    vec = (vec - mean) / std

    return kmeans.predict([vec])[0]

def hand_to_ids(card1, card2) -> str:
    if card1[0] == card2[0]:
        return str(card1[0]) + str(card2[0])
    else:
        r1 = card1[0] if max(RANKS.index(card1[0]), RANKS.index(card2[0])) == RANKS.index(card1[0]) else card2[0]
        r2 = card1[0] if r1 == card2[0] else card2[0]

        if card1[1] == card2[1]:
            return r1 + r2 + 's'
        else: #card1[1] != card2[1]
            return r1 + r2 + 'o'

def preflop_loader(preflop_file):
    with open(preflop_file, "rb") as f:
        file = pickle.load(f)
    return file

def preflop_metrics(card1, card2, path):
    hand_id = hand_to_ids(card1, card2)
    data = preflop_loader(path)
    return np.array(list(data[hand_id].values()))

def bucket_loader(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def flop_metrics(hand, board, path):
    abstraction = bucket_loader(path / "flop_abstraction.pkl")
    metrics = bucket_loader(path / "flop_bucket_metrics.pkl")

    bucket_id = get_flop_bucket(board, abstraction)

    bucket_data = metrics[bucket_id]

    vec = np.array([
        bucket_data["range_metrics"]["RangeAdv"],
        bucket_data["range_metrics"]["NutAdv"],
        bucket_data["range_metrics"]["EPI"],
        bucket_data["range_metrics"]["ECI"],
        bucket_data["range_metrics"]["ShowdownDensity"],
        bucket_data["range_metrics"]["LockIn"],
    ], dtype=np.float32)

    return vec

def get_turn_bucket(board, abstraction):
    kmeans = abstraction["kmeans"]
    mean = abstraction["mean"]
    std = abstraction["std"]

    cards = [Card.new(to_treys(c)) for c in board[:4]]
    c1, c2, c3, c4 = cards

    vec = extract_turn_board_features(c1, c2, c3, c4)
    vec = (vec - mean) / std

    return kmeans.predict([vec])[0]

def turn_metrics(hand, board, path):
    abstraction = bucket_loader(path / "turn_abstraction.pkl")
    metrics = bucket_loader(path / "turn_bucket_metrics.pkl")

    bucket_id = get_turn_bucket(board, abstraction)

    bucket_data = metrics[bucket_id]

    vec = np.array([
        bucket_data["range_metrics"]["RangeAdv"],
        bucket_data["range_metrics"]["NutAdv"],
        bucket_data["range_metrics"]["EPI"],
        bucket_data["range_metrics"]["ECI"],
        bucket_data["range_metrics"]["ShowdownDensity"],
        bucket_data["range_metrics"]["LockIn"],
    ], dtype=np.float32)

    return vec

def encode_board(board: list[str]) -> np.ndarray:
    pass
    '''padded = board + ["??"] * (5 - len(board))
    vec = np.zeros(52, dtype=np.float32)
    for c in padded:
        if c != "??":
            vec[card_to_index(c)] = 1.0
    return vec'''

def encode_street(street: int) -> np.ndarray:
    vec = np.zeros(4, dtype=np.float32)
    vec[street] = 1.0
    return vec

def norm(x, max_stack):
    return np.array([x / max_stack], dtype=np.float32)

def legal_action_mask(legal_actions):
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for a in legal_actions:
        mask[a] = 1.0
    return mask

import numpy as np

def encode_observation(self, player):
    opp = self.p1 if player is self.p2 else self.p2
    max_stack = self.initial_stack * 2

    DATA_PATH = Path.cwd().parents[1] / "data"

    # PREFLOP
    if self.street == 0:
        features = preflop_metrics(
            player.hand[0],
            player.hand[1],
            DATA_PATH / 'preflop_metrics.pkl'
        )

    # FLOP
    elif self.street == 1:
        features = flop_metrics(
            player.hand,
            self.board,
            DATA_PATH
        )

    # TURN
    elif self.street == 2:
        features = turn_metrics(
            player.hand,
            self.board,
            DATA_PATH
        )

    # RIVER
    else:
        features = turn_metrics(
            player.hand,
            self.board,
            DATA_PATH
        )

    obs = np.concatenate([
        features,

        encode_street(self.street),

        norm(player.stack, max_stack),
        norm(opp.stack, max_stack),
        norm(self.engine.pot, max_stack),
        norm(self.engine.to_call, max_stack),

        np.array([
            1.0 if player.position == "SB" else 0.0,
            1.0 if player.position == "BB" else 0.0
        ], dtype=np.float32),

        legal_action_mask(self.legal_actions())
    ])

    return obs
