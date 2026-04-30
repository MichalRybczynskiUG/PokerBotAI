from pathlib import Path
import pickle
import numpy as np
import torch

from src.poker_enviroment.constants import NUM_ACTIONS, RANKS, SUITS
from treys import Card
from src.game_abstraction.board_features import extract_flop_board_features
from src.game_abstraction.board_features import extract_turn_board_features


def get_flop_bucket(board, abstraction):
    kmeans = abstraction["kmeans"]
    mean = abstraction["mean"]
    std = abstraction["std"]

    c1, c2, c3 = [Card.new(c) for c in board]

    vec = extract_flop_board_features(c1, c2, c3)
    vec = (vec - mean) / std

    return kmeans.predict([vec])[0]


def get_turn_bucket(board, abstraction):
    kmeans = abstraction["kmeans"]
    mean = abstraction["mean"]
    std = abstraction["std"]

    cards = [Card.new(c) for c in board[:4]]
    c1, c2, c3, c4 = cards

    vec = extract_turn_board_features(c1, c2, c3, c4)
    vec = (vec - mean) / std

    return kmeans.predict([vec])[0]

def hand_to_ids(card1, card2) -> str:
    r1, r2 = card1[0], card2[0]

    if RANKS.index(r1) < RANKS.index(r2):
        r1, r2 = r2, r1

    if r1 == r2:
        return r1 + r2

    if card1[1] == card2[1]:
        return r1 + r2 + 's'
    else:
        return r1 + r2 + 'o'

def preflop_loader(preflop_file):
    with open(preflop_file, "rb") as f:
        return pickle.load(f)


def preflop_metrics(card1, card2, path):
    hand_id = hand_to_ids(card1, card2)
    data = preflop_loader(path)
    return np.array(list(data[hand_id].values()), dtype=np.float32)

def bucket_loader(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def flop_metrics(hand, board, path):
    abstraction = bucket_loader(path / "flop_abstraction.pkl")
    metrics = bucket_loader(path / "flop_bucket_metrics.pkl")

    bucket_id = get_flop_bucket(board, abstraction)
    bucket_data = metrics[bucket_id]

    return np.array([
        bucket_data["range_metrics"]["RangeAdv"],
        bucket_data["range_metrics"]["NutAdv"],
        bucket_data["range_metrics"]["EPI"],
        bucket_data["range_metrics"]["ECI"],
        bucket_data["range_metrics"]["ShowdownDensity"],
        bucket_data["range_metrics"]["LockIn"],
    ], dtype=np.float32)


def turn_metrics(hand, board, path):
    abstraction = bucket_loader(path / "turn_abstraction.pkl")
    metrics = bucket_loader(path / "turn_bucket_metrics.pkl")

    bucket_id = get_turn_bucket(board, abstraction)
    bucket_data = metrics[bucket_id]

    return np.array([
        bucket_data["range_metrics"]["RangeAdv"],
        bucket_data["range_metrics"]["NutAdv"],
        bucket_data["range_metrics"]["EPI"],
        bucket_data["range_metrics"]["ECI"],
        bucket_data["range_metrics"]["ShowdownDensity"],
        bucket_data["range_metrics"]["LockIn"],
    ], dtype=np.float32)

def encode_street(street: int):
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

def encode_observation(self, player):
    opp = self.p1 if player is self.p2 else self.p2
    max_stack = self.initial_stack * 2

    if self.street == 0:
        hand_id = hand_to_ids(player.hand[0], player.hand[1])
        data = self.preflop_data[hand_id]


        features = np.array([
            data["EHS"],
            #data["VAR"],
            #data["MED"],
            #data["IQR"],
            data["NEG_POT"],
            #data["POS_POT"],

            #0.0,  # RangeAdv
            #0.0,  # NutAdv
            #0.0,  # EPI
            #0.0,  # ECI
            #0.0,  # ShowdownDensity
            #0.0,  # LockIn
        ], dtype=np.float32)

    elif self.street == 1:
        bucket_id = get_flop_bucket(self.board, self.flop_abstraction)
        bucket_data = self.flop_metrics[bucket_id]

        hand_key = hand_to_ids(player.hand[0], player.hand[1])
        hand_data = bucket_data["hand_metrics"][hand_key]

        features = np.array([
            hand_data["EHS"],
            #hand_data["VAR"],
            #hand_data["MED"],
            #hand_data["IQR"],
            hand_data["NEG_POT"],
           #hand_data["POS_POT"],

            #bucket_data["range_metrics"]["RangeAdv"],
            #bucket_data["range_metrics"]["NutAdv"],
            #bucket_data["range_metrics"]["EPI"],
            #bucket_data["range_metrics"]["ECI"],
            #bucket_data["range_metrics"]["ShowdownDensity"],
            #bucket_data["range_metrics"]["LockIn"],
        ], dtype=np.float32)

    else:
        bucket_id = get_turn_bucket(self.board, self.turn_abstraction)
        bucket_data = self.turn_metrics[bucket_id]

        hand_key = hand_to_ids(player.hand[0], player.hand[1])
        hand_data = bucket_data["hand_metrics"][hand_key]

        if hand_data is None:
            hand_data = {
                "EHS": 0.5,
                "VAR": 0.0,
                "MED": 0.5,
                "IQR": 0.0,
                "NEG_POT": 0.5,
                "POS_POT": 0.5,
            }

        features = np.array([
            hand_data["EHS"],
            #hand_data["VAR"],
            #hand_data["MED"],
            #hand_data["IQR"],
            hand_data["NEG_POT"],
            #hand_data["POS_POT"],

            #bucket_data["range_metrics"]["RangeAdv"],
            #bucket_data["range_metrics"]["NutAdv"],
            #bucket_data["range_metrics"]["EPI"],
            #bucket_data["range_metrics"]["ECI"],
            #bucket_data["range_metrics"]["ShowdownDensity"],
            #bucket_data["range_metrics"]["LockIn"],
        ], dtype=np.float32)

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

    return torch.tensor(obs, dtype=torch.float32)