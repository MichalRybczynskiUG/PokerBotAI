from pathlib import Path
import pickle
import numpy as np
from treys import Card
from src.poker_enviroment.constants import NUM_ACTIONS, RANKS, SUITS
from src.game_abstraction.board_features import extract_flop_board_features
from src.game_abstraction.board_features import extract_turn_board_features


#======= Preflop functions ===========
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

# ======== Flop functions =============

def load_flop_runtime(board_file,abstraction_file):
    with open(abstraction_file, "rb") as f:
        abstraction = pickle.load(f)

    with open(board_file, "rb") as f:
        bucket_data = pickle.load(f)

    return abstraction, bucket_data

def flop_to_bucket(card1, card2, card3, abstraction):
    vec = extract_flop_board_features(card1, card2, card3)
    vec_norm = (vec - abstraction["mean"]) / abstraction["std"]
    return int(abstraction["kmeans"].predict(vec_norm.reshape(1, -1))[0])


def get_flop_info(card1, card2, card3, abstraction, bucket_data):
    bucket = flop_to_bucket(card1, card2, card3, abstraction)

    info = bucket_data.get(bucket, None)
    if info is None:
        raise ValueError(f"No data for bucket {bucket}")

    return {
        "bucket": bucket,
        "range_metrics": info["range_metrics"],
        "hand_metrics": info["hand_metrics"]
    }

# =========== Turn functions ===========
def load_turn_runtime(board_file,abstraction_file):
    with open(abstraction_file, "rb") as f:
        abstraction = pickle.load(f)

    with open(board_file, "rb") as f:
        bucket_data = pickle.load(f)

    return abstraction, bucket_data

def turn_to_bucket(card1, card2, card3,card4, abstraction):
    vec = extract_turn_board_features(card1, card2, card3,card4)
    vec_norm = (vec - abstraction["mean"]) / abstraction["std"]
    return int(abstraction["kmeans"].predict(vec_norm.reshape(1, -1))[0])

def get_turn_info(card1, card2, card3,card4, abstraction, bucket_data):
    bucket = turn_to_bucket(card1, card2, card3, card4, abstraction)

    info = bucket_data.get(bucket, None)
    if info is None:
        raise ValueError(f"No data for bucket {bucket}")

    return {
        "bucket": bucket,
        "range_metrics": info["range_metrics"],
        "hand_metrics": info["hand_metrics"]
    }

# =========== Other info ===============

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

def encode_observation(self, player):
    opp = self.p1 if player is self.p2 else self.p2

    max_stack = self.initial_stack * 2

    data_path = Path.cwd().parents[1] / "data"
    street_metrics = np.zeros(6, dtype=np.float32)
    hand = hand_to_ids(player.hand[0], player.hand[1])

    if self.street == 0: #preflop
        file_name = 'preflop_metrics.pkl'
        street_metrics = preflop_metrics(player.hand[0],player.hand[1], data_path / file_name)

    elif self.street == 1: #flop
        abstraction, bucket_data = load_flop_runtime(data_path / "flop_bucket_metrics.pkl",
                                                     data_path / "flop_abstraction.pkl")
        info = get_flop_info(
            Card.new(self.board[0]),
            Card.new(self.board[1]),
            Card.new(self.board[2]),
            abstraction,
            bucket_data
        )

        hand_metric = np.array(list(info["hand_metrics"][hand].values()))
        range_metrics = np.array(list(info["range_metrics"].values()))

        street_metrics = np.concatenate([hand_metric, range_metrics])

    elif self.street == 2: #preflop
        abstraction, bucket_data = load_turn_runtime(data_path / "turn_bucket_metrics.pkl",
                                                     data_path / "turn_abstraction.pkl")

        info = get_turn_info(
            Card.new(self.board[0]),
            Card.new(self.board[1]),
            Card.new(self.board[2]),
            Card.new(self.board[3]),
            abstraction,
            bucket_data
        )

        hand_metric = np.array(list(info["hand_metrics"][hand].values()))
        range_metrics = np.array(list(info["range_metrics"].values()))

        street_metrics = np.concatenate([hand_metric, range_metrics])

    elif self.street == 3: #river
        abstraction, bucket_data = load_turn_runtime(data_path / "turn_bucket_metrics.pkl",
                                                     data_path / "turn_abstraction.pkl")

        info = get_turn_info(
            Card.new(self.board[1]),
            Card.new(self.board[2]),
            Card.new(self.board[3]),
            Card.new(self.board[4]),
            abstraction,
            bucket_data
        )

        hand_metric = np.array(list(info["hand_metrics"][hand].values()))
        range_metrics = np.array(list(info["range_metrics"].values()))

        street_metrics = np.concatenate([hand_metric, range_metrics])


    obs = np.concatenate([
        street_metrics,          #
        encode_street(self.street),             # 4

        norm(player.stack, max_stack),           # 1
        norm(opp.stack, max_stack),              # 1
        norm(self.engine.pot, max_stack),        # 1
        norm(self.engine.to_call, max_stack),    # 1

        np.array([1.0 if player.position == "SB" else 0.0]),  # 1

        legal_action_mask(self.legal_actions())  # 4
    ])


    return obs
