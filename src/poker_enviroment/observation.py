import pickle
import numpy as np
import torch

from src.poker_enviroment.constants import (
    NUM_ACTIONS,
    RANKS,
    SUITS,
)

def encode_round_cards(hand, board):
    vecs = []

    vecs.append(encode_cards(hand))

    vecs.append(encode_cards(board[:3]))

    vecs.append(encode_cards(board[:4]))

    vecs.append(encode_cards(board[:5]))

    return np.concatenate(vecs)  # 208

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

def bucket_loader(path):
    with open(path, "rb") as f:
        return pickle.load(f)

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

def encode_cards(cards):
    vec = np.zeros(52, dtype=np.float32)

    for c in cards:
        rank = RANKS.index(c[0])
        suit = SUITS.index(c[1])
        idx = rank * 4 + suit
        vec[idx] = 1.0

    return vec

def encode_observation(self, player):
    opp = self.p1 if player is self.p2 else self.p2
    max_stack = self.initial_stack * 2

    cards_vec = encode_round_cards(player.hand, self.board)

    stack_self = norm(player.stack, max_stack)
    stack_opp = norm(opp.stack, max_stack)
    pot = norm(self.engine.pot, max_stack)

    to_call_amount = max(0, self.engine.to_call - player.street_bet)
    to_call = norm(to_call_amount, max_stack)

    position = np.array([
        1.0 if player.position == "SB" else 0.0,
        1.0 if player.position == "BB" else 0.0
    ], dtype=np.float32)

    history_vec = self.action_history.flatten()

    obs = np.concatenate([
        cards_vec,
        history_vec,
        stack_self,
        stack_opp,
        pot,
        to_call,
        position
    ])

    return torch.tensor(obs, dtype=torch.float32)