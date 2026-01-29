import numpy as np
from constants import NUM_ACTIONS, RANKS, SUITS

def card_to_index(card: str) -> int:
    rank = card[0]
    suit = card[1]
    return RANKS.index(rank) * 4 + SUITS.index(suit)

def encode_cards(cards: list[str]) -> np.ndarray:
    vec = np.zeros(52, dtype=np.float32)
    for c in cards:
        vec[card_to_index(c)] = 1.0
    return vec

def encode_board(board: list[str]) -> np.ndarray:
    padded = board + ["??"] * (5 - len(board))
    vec = np.zeros(52, dtype=np.float32)
    for c in padded:
        if c != "??":
            vec[card_to_index(c)] = 1.0
    return vec

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

    obs = np.concatenate([
        encode_cards(player.hand),             # 52
        encode_board(self.board),               # 52
        encode_street(self.street),             # 4

        norm(player.stack, max_stack),           # 1
        norm(opp.stack, max_stack),              # 1
        norm(self.engine.pot, max_stack),        # 1
        norm(self.engine.to_call, max_stack),    # 1

        np.array([1.0 if player.position == "SB" else 0.0]),  # 1

        legal_action_mask(self.legal_actions())  # 4
    ])

    return obs
