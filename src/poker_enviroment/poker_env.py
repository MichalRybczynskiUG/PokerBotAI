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


def play_hand(p1, p2, dealer) -> None:

    # Ustal pozycje
    if dealer == p1:
        sb = p1
        bb = p2
    else:
        sb = p2
        bb = p1

    sb.position = "SB / BTN"
    bb.position = "BB"

    # Przygotuj talię
    deck = create_deck()
    random.shuffle(deck)

    # Reset
    for p in (p1, p2):
        p.reset()

    pot = 0

    # --- Blindy ---
    sb_amount = min(SMALL_BLIND, sb.stack)
    bb_amount = min(BIG_BLIND, bb.stack)

    sb.stack -= sb_amount
    bb.stack -= bb_amount

    sb.bet = sb_amount
    bb.bet = bb_amount

    sb.street_bet = sb_amount
    bb.street_bet = bb_amount

    pot += sb_amount + bb_amount

    to_call = bb.bet

    print(f"{sb.name} ({sb.position}) stawia SB = {sb_amount}")
    print(f"{bb.name} ({bb.position}) stawia BB = {bb_amount}")


    # Rozdaj karty
    p1.hand = [deal_card(deck), deal_card(deck)]
    p2.hand = [deal_card(deck), deal_card(deck)]

    print("\n=== Rozdanie ===")

    # Preflop
    print("\n=== Preflop ===")
    pot, to_call = betting_round([sb, bb], pot, to_call)

    if p1.folded or p2.folded:
        winner = p1 if not p1.folded else p2
        winner.stack += pot
        print(f"\n{winner.name} zgarnia pulę {pot} (drugi spasował).")
        return

    # Flop
    board = [deal_card(deck), deal_card(deck), deal_card(deck)]
    print("\n=== Flop ===")
    print("Board:", board)
    for p in (p1, p2):
        p.street_bet = 0
    pot, to_call = betting_round([bb, sb], pot, 0)

    if p1.folded or p2.folded:
        winner = p1 if not p1.folded else p2
        winner.stack += pot
        print(f"\n{winner.name} zgarnia pulę {pot} (drugi spasował).")
        return

    # Turn
    board.append(deal_card(deck))
    print("\n=== Turn ===")
    print("Board:", board)
    for p in (p1, p2):
        p.street_bet = 0
    pot, to_call = betting_round([bb, sb], pot, 0)

    if p1.folded or p2.folded:
        winner = p1 if not p1.folded else p2
        winner.stack += pot
        print(f"\n{winner.name} zgarnia pulę {pot} (drugi spasował).")
        return

    # River
    board.append(deal_card(deck))
    print("\n=== River ===")
    print("Board:", board)
    for p in (p1, p2):
        p.street_bet = 0
    pot, to_call = betting_round([bb, sb], pot, 0)

    if p1.folded or p2.folded:
        winner = p1 if not p1.folded else p2
        winner.stack += pot
        print(f"\n{winner.name} zgarnia pulę {pot} (drugi spasował).")
        return

    # Showdown
    print("\n=== Showdown ===")

    from src.PokerEnv.hand_eval import evaluate_hand

    score1 = evaluate_hand(p1.hand, board)
    score2 = evaluate_hand(p2.hand, board)

    print(f"{p1.name}: {p1.hand}, wynik = {score1}")
    print(f"{p2.name}: {p2.hand}, wynik = {score2}")

    # --- SIDE POTS ---
    pots = build_pots([p1, p2])

    print("\nPule:")
    for i, pot_info in enumerate(pots):
        names = ", ".join(p.name for p in pot_info["eligible"])
        print(f"  Pula {i + 1}: {pot_info['amount']} (gracze: {names})")

    # oceny rąk
    scores = {
        p1: score1,
        p2: score2
    }

    # rozdzielanie pul
    for pot_info in pots:
        eligible = [p for p in pot_info["eligible"] if not p.folded]

        # wybieramy zwycięzcę puli
        best_score = max(scores[p] for p in eligible)
        winners = [p for p in eligible if scores[p] == best_score]

        share = pot_info["amount"] // len(winners)
        remainder = pot_info["amount"] % len(winners)

        for w in winners:
            w.stack += share
            print(f"{w.name} wygrywa {share} z tej puli.")

        if remainder > 0:
            winners[0].stack += remainder


def main():
    print("Minimalny Poker NL Hold’em — 1v1")

    p1_name = input("Podaj nazwę gracza 1: ")
    p2_name = input("Podaj nazwę gracza 2: ")

    p1 = Player(p1_name)
    p2 = Player(p2_name)

    dealer = p1  # p1 zaczyna jako dealer

    while True:
        play_hand(p1, p2, dealer)

        # zmiana dealera
        dealer = p2 if dealer == p1 else p1

        print(f"\nStacki: {p1.name}: {p1.stack} | {p2.name}: {p2.stack}")

        cont = input("\nCzy rozdać kolejną rękę? (t/n): ").lower()
        if cont != "t":
            print("Dzięki za grę!")
            break


if __name__ == "__main__":
    main()