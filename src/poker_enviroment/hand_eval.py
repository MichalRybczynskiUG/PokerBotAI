import eval7

def evaluate_hand(hand, board):
    cards = [eval7.Card(c) for c in hand + board]
    return eval7.evaluate(cards)