from src.poker_enviroment.poker_env import create_deck
from src.poker_enviroment.observation import hand_to_ids

class TestCreateDeck:
    def test_create_deck_length(self):
        deck = create_deck()
        assert len(deck) == 52

    def test_create_deck_unique(self):
        deck = create_deck()
        assert len(deck) == len(set(deck))

class TestObservation:
    def test_hand_to_ids(self):
        hand_id = hand_to_ids('4♣', '4♦')
        assert len(hand_id) == 2

    def test_hand_to_ids_2(self):
        hand_id = hand_to_ids('2♣', '4♣')
        assert hand_id[2] == 's'

    def test_hand_to_ids_3(self):
        hand_id = hand_to_ids('2♥', '4♣')
        assert hand_id[2] == 'o'

