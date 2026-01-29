from pathlib import Path
import pickle
import numpy as np
from src.poker_enviroment.constants import NUM_ACTIONS, RANKS, SUITS

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
    if self.street == 0:
        obs = np.concatenate([
            preflop_metrics(player.hand[0],player.hand[1], DATA_PATH / 'preflop_metrics.pkl'),             # 6
            #encode_board(self.board),               # 52
            encode_street(self.street),             # 4

            norm(player.stack, max_stack),           # 1
            norm(opp.stack, max_stack),              # 1
            norm(self.engine.pot, max_stack),        # 1
            norm(self.engine.to_call, max_stack),    # 1

            np.array([1.0 if player.position == "SB" else 0.0]),  # 1

            legal_action_mask(self.legal_actions())  # 4
        ])
    else:
        obs = np.concatenate([])

    return obs
