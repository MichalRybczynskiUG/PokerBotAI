import random
import itertools

SUITS = ["♠", "♥", "♦", "♣"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SMALL_BLIND = 5
BIG_BLIND = 10
MIN_RAISE = BIG_BLIND


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
        actions.append("f")

    # Call tylko jeśli jest co callować
    if to_call > player.street_bet and player.stack > 0:
        actions.append("c")

    # Check tylko jeśli nic nie trzeba dopłacać
    if to_call == player.street_bet:
        actions.append("c") #check

    # Raise tylko jeśli gracz ma środki
    if player.stack > (to_call - player.street_bet):
        actions.append("r")

    # All-in zawsze możliwy jeśli gracz ma stack
    if player.stack > 0:
        actions.append("a")

    return actions

def betting_finished(players, actions_without_raise, players_in_hand) -> bool:

    for p in players:
        if p.folded:
            continue
        #if not p.all_in and p.street_bet != to_call:
        if not p.all_in and actions_without_raise >= len(players_in_hand):
            return False
    return True

def betting_round(players, pot, to_call) -> tuple[int, int]:

    actions_without_raise = 0
    players_in_hand = [p for p in players if not p.folded and not p.all_in]

    # Jeśli wszyscy są all-in lub fold → brak licytacji
    if all(p.all_in or p.folded for p in players):
        return pot, to_call

    idx = 0  # indeks aktualnego gracza

    while True:
        p = players[idx % len(players)]
        idx += 1

        if p.folded or p.all_in:
            # pomijamy graczy bez akcji
            if betting_finished(players, actions_without_raise, players_in_hand):
                break
            continue

        print(f"\n--- Tura gracza {p.name} ({p.position}) ---")
        print(f"Karty: {p.hand[0]}  {p.hand[1]}")
        print(f"Stack: {p.stack}")
        print(f"Pot: {pot}")
        print(f"Do sprawdzenia: {to_call - p.street_bet}")
        print(f"to_call: {to_call}")
        print(f"p.street_bet: {p.street_bet}")

        legal_actions = get_legal_actions(p, to_call)
        print("Akcje:", ", ".join(legal_actions))

        while True:
            action = input("Wybierz akcję: ").strip().lower()
            if action in legal_actions:
                break
            print("Nielegalna akcja. Spróbuj ponownie.")

        # --- FOLD ---
        if action == "f":
            p.folded = True
            print(f"{p.name} spasował.")
            return pot, to_call

        # --- CALL / CHECK ---
        elif action == "c":
            call_amount = max(0, to_call - p.street_bet) # call_amount nie może być ujemne
            call_amount = min(call_amount, p.stack) # call_amount nie może być większe niż stack

            p.stack -= call_amount
            p.bet += call_amount
            p.street_bet += call_amount
            pot += call_amount
            actions_without_raise += 1

            if p.stack == 0:
                p.all_in = True
                print(f"{p.name} sprawdził i jest all-in!")

        # --- RAISE ---
        elif action == "r":
            while True:
                try:
                    raise_amount = int(input(f"Kwota raise (min {MIN_RAISE}): "))
                    if raise_amount < MIN_RAISE:
                        print("Raise za mały.")
                        continue
                    break
                except:
                    print("Podaj liczbę.")

            call_amount = max(0, to_call - p.street_bet)
            total = call_amount + raise_amount

            if total >= p.stack:
                print(f"{p.name} przebija all-in!")
                pot += p.stack
                p.bet += p.stack
                p.street_bet += p.stack
                p.stack = 0
                p.all_in = True
            else:
                p.stack -= total
                p.bet += total
                p.street_bet += total
                pot += total

            #to_call = p.bet
            to_call = p.street_bet
            actions_without_raise = 0
            continue  #po raise NIE sprawdzamy końca rundy

        # --- ALL-IN ---
        elif action == "a":
            all_in_amount = p.stack
            print(f"{p.name} wchodzi all-in za {all_in_amount}!")



            p.stack = 0
            p.bet += all_in_amount
            p.street_bet += all_in_amount
            pot += all_in_amount
            p.all_in = True

            if p.bet > to_call:
                #to_call = p.bet
                to_call = p.street_bet
                actions_without_raise = 0
                continue  # all-in jako raise → kolejka od nowa

        # Po akcji sprawdzamy, czy runda może się skończyć
        if betting_finished(players, actions_without_raise, players_in_hand):
            break

        #if actions_without_raise >= len(players_in_hand):
        #    break

    return pot, to_call

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