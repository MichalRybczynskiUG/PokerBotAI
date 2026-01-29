from poker_enviroment.poker_env import create_deck
from poker_enviroment.observation import encode_cards

class TestCreateDeck:
    def test_create_deck_length(self):
        deck = create_deck()
        assert len(deck) == 52

    def test_create_deck_unique(self):
        deck = create_deck()
        assert len(deck) == len(set(deck))

class TestObservation:
    def test_card_encoding(self):
        obs = encode_cards(["As", "Kd"])
        assert obs.sum() == 2
