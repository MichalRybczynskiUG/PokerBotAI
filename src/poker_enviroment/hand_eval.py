import itertools

RANKS = "23456789TJQKA"
RANK_TO_VALUE = {r: i for i, r in enumerate(RANKS)}

def card_value(card):
    return RANK_TO_VALUE[card[0]]

def is_straight(values):
    """Sprawdza czy lista wartości tworzy strita."""
    values = sorted(set(values))
    # specjalny przypadek: A2345
    if values == [0, 1, 2, 3, 12]:
        return True, 3  # piątka (5) jako najwyższa
    if len(values) < 5:
        return False, None

    for i in range(len(values) - 4):
        window = values[i:i+5]
        if window == list(range(window[0], window[0] + 5)):
            return True, window[-1]

    return False, None

def group_by_rank(cards):
    """Zwraca słownik: rank -> [karty]"""
    groups = {}
    for c in cards:
        groups.setdefault(c[0], []).append(c)
    return groups

def evaluate_hand(hand, board):
    """
    Zwraca ranking układu w formie tuple:
    (układ, wartości tiebreak)
    gdzie wyższy tuple oznacza lepszą rękę.
    """

    cards = hand + board
    values = sorted([card_value(c) for c in cards], reverse=True)
    #suits = [c[1] for c in cards]
    groups = group_by_rank(cards)

    # --- Flush ---
    suit_groups = {}
    for c in cards:
        suit_groups.setdefault(c[1], []).append(c)

    flush_cards = None
    for s, cs in suit_groups.items():
        if len(cs) >= 5:
            flush_cards = sorted(cs, key=lambda x: card_value(x), reverse=True)

    # --- Straight ---
    has_straight, straight_high = is_straight([card_value(c) for c in cards])

    # --- Straight Flush ---
    if flush_cards is not None:
        vals = [card_value(c) for c in flush_cards]
        sf, high = is_straight(vals)
        if sf:
            # Royal Flush = straight flush z najwyższym A
            if high == RANK_TO_VALUE['A']:
                return (9, high)  # 9 = royal flush
            return (8, high)      # 8 = straight flush

    # --- Four of a Kind ---
    four = [r for r, cs in groups.items() if len(cs) == 4]
    if four:
        four_rank = RANK_TO_VALUE[four[0]]
        kicker = max(v for v in values if v != four_rank)
        return (7, four_rank, kicker)

    # --- Full House ---
    three = sorted([r for r, cs in groups.items() if len(cs) == 3],
                   key=lambda r: RANK_TO_VALUE[r],
                   reverse=True)
    pairs = sorted([r for r, cs in groups.items() if len(cs) == 2],
                   key=lambda r: RANK_TO_VALUE[r],
                   reverse=True)

    if three and (pairs or len(three) > 1):
        top_three = RANK_TO_VALUE[three[0]]
        top_pair = RANK_TO_VALUE[pairs[0]] if pairs else RANK_TO_VALUE[three[1]]
        return (6, top_three, top_pair)

    # --- Flush ---
    if flush_cards is not None:
        top_five = [card_value(c) for c in flush_cards[:5]]
        return (5, top_five)

    # --- Straight ---
    if has_straight:
        return (4, straight_high)

    # --- Three of a Kind ---
    if three:
        t = RANK_TO_VALUE[three[0]]
        kickers = [v for v in values if v != t][:2]
        return (3, t, kickers)

    # --- Two Pair ---
    if len(pairs) >= 2:
        p1 = RANK_TO_VALUE[pairs[0]]
        p2 = RANK_TO_VALUE[pairs[1]]
        kicker = [v for v in values if v not in (p1, p2)][0]
        return (2, p1, p2, kicker)

    # --- One Pair ---
    if pairs:
        p = RANK_TO_VALUE[pairs[0]]
        kickers = [v for v in values if v != p][:3]
        return (1, p, kickers)

    # --- High Card ---
    return (0, values[:5])

