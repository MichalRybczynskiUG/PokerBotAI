import random
import itertools

SUITS = ["♠", "♥", "♦", "♣"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SMALL_BLIND = 5
BIG_BLIND = 10
MIN_RAISE = BIG_BLIND

ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_RAISE = 2
ACTION_ALL_IN = 3

ACTION_NAMES = {
    0: "fold",
    1: "call/check",
    2: "raise",
    3: "all-in"
}

def create_deck() -> list[str]:
    return [r + s for r, s in itertools.product(RANKS, SUITS)]

def deal_card(deck) -> None:
    return deck.pop()

class Player:
    def __init__(self, name, stack=1000):
        self.name = name
        self.stack = stack
        self.hand = []
        self.folded = False
        self.all_in = False
        self.position = None
        self.bet = 0
        self.street_bet = 0

    def reset(self):
        self.hand = []
        self.folded = False
        self.bet = 0
        self.all_in = False
        self.street_bet = 0

def get_legal_actions(player, to_call) -> list[str]:

    actions = []

    # Fold zawsze możliwy (jeśli jest bet)
    if to_call > player.street_bet:
        actions.append(ACTION_FOLD)

    # Call tylko jeśli jest co callować
    if to_call > player.street_bet and player.stack > 0:
        actions.append(ACTION_CALL)

    # Check tylko jeśli nic nie trzeba dopłacać
    if to_call == player.street_bet:
        actions.append(ACTION_CALL) #check

    # Raise tylko jeśli gracz ma środki
    if player.stack > (to_call - player.street_bet):
        actions.append(ACTION_RAISE)

    # All-in zawsze możliwy jeśli gracz ma stack
    if player.stack > 0:
        actions.append(ACTION_ALL_IN)

    return actions


class PokerEngine:
    def __init__(self, players):
        self.players = players
        self.pot = 0
        self.to_call = 0
        self.current_player_idx = 0
        self.actions_without_raise = 0

    def next_player(self):
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

    def betting_round_finished(self):
        active = [p for p in self.players if not p.folded and not p.all_in]

        if len(active) <= 1:
            return True

        if self.actions_without_raise >= len(active):
            return True

        return False

    def step_betting(self, action, raise_amount=None):
        p = self.players[self.current_player_idx]

        if p.folded or p.all_in:
            self.next_player()
            return

        legal = get_legal_actions(p, self.to_call)
        if action not in legal:
            raise ValueError("Illegal action")

        # --- FOLD ---
        if action == ACTION_FOLD:
            p.folded = True

        # --- CALL / CHECK ---
        elif action == ACTION_CALL:
            call_amount = max(0, self.to_call - p.street_bet)  # call_amount nie może być ujemne
            call_amount = min(call_amount, p.stack)  # call_amount nie może być większe niż stack

            p.stack -= call_amount
            p.bet += call_amount
            p.street_bet += call_amount
            self.pot += call_amount
            self.actions_without_raise += 1

            if p.stack == 0:
                p.all_in = True

        # --- RAISE ---
        elif action == ACTION_RAISE:
            if raise_amount is None:
                raise ValueError("Raise requires raise_amount")

            call_amount = max(0, self.to_call - p.street_bet)
            total = call_amount + raise_amount

            if total >= p.stack:
                total = p.stack
                p.all_in = True

            p.stack -= total
            p.bet += total
            p.street_bet += total
            self.pot += total

            self.to_call = p.street_bet
            self.actions_without_raise = 0
            # po raisie nie sprawdzamy końca  rundy

        # --- ALL-IN ---
        elif action == ACTION_ALL_IN:
            amount = p.stack
            p.stack = 0
            p.bet += amount
            p.street_bet += amount
            self.pot += amount
            p.all_in = True

            if p.street_bet > self.to_call:
                self.to_call = p.street_bet
                self.actions_without_raise = 0

        self.next_player()


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


PREFLOP = 0
FLOP = 1
TURN = 2
RIVER = 3
SHOWDOWN = 4

class PokerEnv:
    def __init__(self, stack_size=1000):
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

        prev_stack = self.current_player.stack

        self.engine.step_betting(action, raise_amount)

        # aktualizacja current_player
        self.current_player = self.engine.players[self.engine.current_player_idx]

        # sprawdzamy koniec rundy
        if self.engine.betting_round_finished():
            self._advance_street()

        reward = self.current_player.stack - prev_stack
        obs = self._get_observation(self.current_player)

        return obs, reward, self.done, {}

    def _advance_street(self):
        for p in self.players:
            p.street_bet = 0

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

        self.engine.actions_without_raise = 0
        self.engine.to_call = 0

    def _showdown(self):
        from PokerBotAI.src.poker_enviroment.hand_eval import evaluate_hand

        # ocena rąk TYLKO raz
        scores = {
            p: evaluate_hand(p.hand, self.board)
            for p in self.players
            if not p.folded
        }

        pots = build_pots(self.players)

        for pot in pots:
            amount = pot["amount"]

            # tylko gracze:
            # 1) którzy są eligible do tej puli
            # 2) którzy nie spasowali
            eligible = [
                p for p in pot["eligible"]
                if not p.folded
            ]

            if not eligible:
                continue  # teoretycznie nie powinno się zdarzyć

            # najlepsza ręka w tej puli
            best_score = max(scores[p] for p in eligible)

            winners = [
                p for p in eligible
                if scores[p] == best_score
            ]

            share = amount // len(winners)
            remainder = amount % len(winners)

            for w in winners:
                w.stack += share

            # reszta żetonów (odd chip) — standardowo pierwszy gracz
            if remainder > 0:
                winners[0].stack += remainder

        self.done = True

    def _get_observation(self, player):
        opp = self.p1 if player is self.p2 else self.p2

        return {
            "hand": player.hand,
            "board": self.board.copy(),
            "stack_self": player.stack,
            "stack_opp": opp.stack,
            "pot": self.engine.pot,
            "to_call": self.engine.to_call,
            "street": self.street,
            "legal_actions": self.legal_actions()
        }


def main():
    env = PokerEnv()
    obs = env.reset()

    

if __name__ == "__main__":
    main()