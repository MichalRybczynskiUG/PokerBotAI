from PokerBotAI.src.poker_enviroment.poker_env import create_deck


class TestCreateDeck:
    def test_create_deck_length(self):
        deck = create_deck()
        assert len(deck) == 52

    def test_create_deck_unique(self):
        deck = create_deck()
        assert len(deck) == len(set(deck))


