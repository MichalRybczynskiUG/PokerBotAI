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