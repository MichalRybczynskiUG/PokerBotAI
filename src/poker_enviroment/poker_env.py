import random
import itertools
from src.poker_enviroment.Player import *
from src.poker_enviroment.poker_engine import *
from src.poker_enviroment.constants import *
from src.poker_enviroment.observation import encode_observation


def create_deck() -> list[str]:
    return [r + s for r, s in itertools.product(RANKS, SUITS)]

def deal_card(deck) -> None:
    return deck.pop()

def build_pots(players) -> list:

    pots = []

    # gracze, którzy włożyli cokolwiek
    active = [p for p in players if p.bet > 0]

    # sortujemy po wysokości betu
    active.sort(key=lambda p: p.bet)

    prev = 0
    while active:
        # najmniejszy bet wśród pozostałych
        level = active[0].bet

        # pula, o którą grają gracze na tym samym level
        amount = (level - prev) * len(active)

        pots.append({
            "amount": amount,
            "eligible": active.copy()
        })

        prev = level

        # usuwamy graczy, którzy „skończyli” na tym poziomie
        active = [p for p in active if p.bet > level]

    return pots


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

    @property
    def initial_stack(self):
        return self._initial_stack

    def reset(self):
        # reset graczy
        for p in self.players:
            p.reset()

        self.board = []
        self.street = PREFLOP
        self.done = False

        # deck
        self.deck = create_deck()
        random.shuffle(self.deck)

        # rozdanie
        for p in self.players:
            p.hand = [deal_card(self.deck), deal_card(self.deck)]

        # blindy
        sb, bb = self.players

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
        self.last_stacks = {p: p.stack for p in self.players}

        return self._get_observation(self.current_player)

    def legal_actions(self):
        return get_legal_actions(self.current_player, self.engine.to_call)

    def step(self, action, raise_amount=None):
        if self.done:
            raise RuntimeError("Hand already finished")

        acting_player = self.current_player
        prev_stack = acting_player.stack


        self.engine.step_betting(action, raise_amount)
        self.current_player = self.engine.players[self.engine.current_player_idx]

        if self.engine.betting_round_finished():
            self._advance_street()
        reward = 0.0

        if self.done:
            p1_stack = self.p1.stack
            p2_stack = self.p2.stack

            if p1_stack > p2_stack:
                reward = 1.0
            elif p2_stack > p1_stack:
                reward = -1.0
            else:
                reward = 0.0

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

        pots = build_pots(self.players)

        for pot in pots:
            amount = pot["amount"]

            eligible = [
                p for p in pot["eligible"]
                if not p.folded
            ]

            if not eligible:
                continue

            best_score = max(scores[p] for p in eligible)

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

        self.done = True

    def _get_observation(self, player):
        return encode_observation(self, player)


