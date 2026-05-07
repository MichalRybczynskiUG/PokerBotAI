import random
import itertools
from src.poker_enviroment.Player import *
from src.poker_enviroment.poker_engine import *
from src.poker_enviroment.constants import *
from src.poker_enviroment.observation import encode_observation
import numpy as np

def map_action_to_bucket(action):
    if action == ACTION_CALL:
        return 0

    elif action in [ACTION_BET_25, ACTION_BET_33]:
        return 1

    elif action == ACTION_BET_50:
        return 2

    elif action in [ACTION_BET_75, ACTION_BET_100]:
        return 3

    elif action == ACTION_ALL_IN:
        return 4

    else:
        return None

def create_deck() -> list[str]:
    return [r + s for r, s in itertools.product(RANKS, SUITS)]

def deal_card(deck) -> None:
    return deck.pop()

def build_pots(players, pot):
    return [{
        "amount": pot,
        "eligible": [p for p in players if not p.folded]
    }]


class PokerEnv:
    def __init__(self, stack_size=1000):
        self._initial_stack = stack_size
        self.p1 = Player("P1", stack_size)
        self.p2 = Player("P2", stack_size)
        self.players = [self.p1, self.p2]

        self.engine = None
        self.deck = None
        self.board = []
        self.street = PREFLOP
        self.done = False

        self.current_player = None
        self.last_stacks = None

        self.action_history = np.zeros((2, 4, 5, 5), dtype=np.float32)


    @property
    def initial_stack(self):
        return self._initial_stack

    def reset(self):

        for p in self.players:
            p.reset()
        self.initial_stack_p1 = self.p1.stack
        self.initial_stack_p2 = self.p2.stack

        self.board = []
        self.street = PREFLOP
        self.done = False

        self.action_history.fill(0)

        # deck
        self.deck = create_deck()
        random.shuffle(self.deck)

        # rozdanie
        for p in self.players:
            p.hand = [deal_card(self.deck), deal_card(self.deck)]

        self.initial_stack_p1 = self.p1.stack
        self.initial_stack_p2 = self.p2.stack

        # blindy
        if random.random() < 0.5:
            sb, bb = self.players
        else:
            bb, sb = self.players

        sb.position = "SB"
        bb.position = "BB"

        sb.stack -= SMALL_BLIND
        bb.stack -= BIG_BLIND

        sb.bet = sb.street_bet = SMALL_BLIND
        bb.bet = bb.street_bet = BIG_BLIND

        self.engine = PokerEngine([sb, bb])
        self.engine.pot = SMALL_BLIND + BIG_BLIND
        self.engine.to_call = BIG_BLIND

        self.current_player = sb

        return self._get_observation(self.current_player)

    def legal_actions(self):
        return get_legal_actions(self.current_player, self.engine.to_call)

    def step(self, action, raise_amount=None):
        if self.done:
            raise RuntimeError("Hand already finished")

        acting_player = self.current_player

        player_idx = 0 if acting_player == self.p1 else 1
        street = self.street

        slot = int(np.sum(self.action_history[player_idx, street].sum(axis=-1) > 0))
        bucket = map_action_to_bucket(action)

        if bucket is not None and slot < 5:
            self.action_history[player_idx, street, slot, bucket] = 1.0

        self.engine.step_betting(action, raise_amount)

        active = [p for p in self.engine.players if not p.folded]

        if len(active) == 1:
            winner = active[0]

            self.done = True

            winner.stack += self.engine.pot
            self.engine.pot = 0

            if acting_player == self.p1:
                reward = (
                                 self.p1.stack - self.initial_stack_p1
                         ) / BIG_BLIND
            else:
                reward = (
                                 self.p2.stack - self.initial_stack_p2
                         ) / BIG_BLIND

            obs = self._get_observation(winner)

            return obs, reward, True, {}

        self.current_player = self.engine.players[self.engine.current_player_idx]

        active = [p for p in self.engine.players if not p.folded]

        if all(p.all_in for p in active):
            while self.street != RIVER:
                self._advance_street()

            self._advance_street()

        if self.engine.betting_round_finished():
            self._advance_street()

        reward = 0.0

        if self.done:

            if acting_player == self.p1:
                reward = (
                                 self.p1.stack - self.initial_stack_p1
                         ) / BIG_BLIND
            else:
                reward = (
                                 self.p2.stack - self.initial_stack_p2
                         ) / BIG_BLIND

        obs = self._get_observation(self.current_player)

        return obs, reward, self.done, {}

    def _advance_street(self):
        for p in self.players:
            p.street_bet = 0

        self.engine.to_call = 0
        self.engine.actions_without_raise = 0

        if self.street == PREFLOP:
            self.board = [deal_card(self.deck) for _ in range(3)]
            self.street = FLOP

        elif self.street == FLOP:
            self.board.append(deal_card(self.deck))
            self.street = TURN

        elif self.street == TURN:
            self.board.append(deal_card(self.deck))
            self.street = RIVER

        elif self.street == RIVER:
            self._showdown()
            return

    def _showdown(self):
        from src.poker_enviroment.hand_eval import evaluate_hand

        scores = {
            p: evaluate_hand(p.hand, self.board)
            for p in self.players
            if not p.folded
        }

        pots = build_pots(self.players, self.engine.pot)

        for pot in pots:
            amount = pot["amount"]

            eligible = [
                p for p in pot["eligible"]
                if not p.folded
            ]

            if not eligible:
                continue

            best_score = min(scores[p] for p in eligible)

            winners = [
                p for p in eligible
                if scores[p] == best_score
            ]

            share = amount // len(winners)
            remainder = amount % len(winners)

            for w in winners:
                w.stack += share

            if remainder > 0:
                winners[0].stack += remainder

        self.engine.pot = 0

        for p in self.players:
            p.bet = 0
            p.street_bet = 0

        self.done = True

    def _get_observation(self, player):
        return encode_observation(self, player)

    def hand_strength(self, player):
        from src.poker_enviroment.hand_eval import evaluate_hand

        score = evaluate_hand(player.hand, self.board)

        return 1.0 / (1.0 + score)

def get_debug_observation(self, player):
    opp = self.p1 if player is self.p2 else self.p2

    return {
        "street": self.street,
        "hand": player.hand,
        "board": self.board,
        "stack_self": player.stack,
        "stack_opp": opp.stack,
        "pot": self.engine.pot,
        "to_call": self.engine.to_call,
        "legal_actions": self.legal_actions()
    }