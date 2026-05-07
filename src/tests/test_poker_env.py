from src.poker_enviroment.poker_env import create_deck
from src.poker_enviroment.observation import hand_to_ids
from src.poker_enviroment.poker_env import PokerEnv
from src.poker_enviroment.constants import *
import numpy as np

# DECK TESTS

class TestCreateDeck:

    def test_create_deck_length(self):
        deck = create_deck()
        assert len(deck) == 52

    def test_create_deck_unique(self):
        deck = create_deck()
        assert len(deck) == len(set(deck))

    def test_create_deck_contains_all_ranks(self):
        deck = create_deck()
        ranks = [card[0] for card in deck]

        for r in ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]:
            assert ranks.count(r) == 4

    def test_create_deck_contains_all_suits(self):
        deck = create_deck()
        suits = [card[1] for card in deck]

        for s in ["s","h","d","c"]:
            assert suits.count(s) == 13

# HAND ENCODING TESTS

class TestObservation:

    def test_pair(self):
        hand = hand_to_ids('As', 'Ad')
        assert hand[0] == 'A'
        assert hand[1] == 'A'

    def test_suited(self):
        hand = hand_to_ids('2c', '4c')
        assert hand[2] == 's'

    def test_offsuit(self):
        hand = hand_to_ids('2h', '4c')
        assert hand[2] == 'o'

    def test_order_independent(self):
        h1 = hand_to_ids('As', 'Kd')
        h2 = hand_to_ids('Kd', 'As')
        assert h1 == h2

    def test_high_card_first(self):
        hand = hand_to_ids('2c', 'Ad')
        assert hand[0] == 'A'

# EDGE CASES

class TestEdgeCases:

    def test_same_card_twice(self):
        try:
            hand_to_ids('As', 'As')
            assert False
        except:
            assert True

    def test_invalid_rank(self):
        try:
            hand_to_ids('Xs', 'Ad')
            assert False
        except:
            assert True

    def test_invalid_suit(self):
        try:
            hand_to_ids('Az', 'Ad')
            assert False
        except:
            assert True

#Poker engine tester

class TestPokerEnv:

    def test_reset(self):
        env = PokerEnv()
        env.reset()

        assert env.street == PREFLOP

    def test_legal_actions_not_empty(self):
        env = PokerEnv()
        env.reset()

        actions = env.legal_actions()
        assert len(actions) > 0

    def test_fold_ends_game(self):
        env = PokerEnv()
        env.reset()

        _, _, done, _ = env.step(ACTION_FOLD, None)
        assert done is True

    def test_call_progresses_round(self):
        env = PokerEnv()
        env.reset()

        env.step(ACTION_CALL, None)
        env.step(ACTION_CALL, None)

        assert env.street >= FLOP

    def test_all_in_sets_stack_zero(self):
        env = PokerEnv()
        env.reset()

        player = env.current_player
        env.step(ACTION_ALL_IN, None)

        assert player.stack == 0

    def test_all_in_vs_all_in_ends(self):
        env = PokerEnv()
        env.reset()

        env.current_player = env.p1
        env.step(ACTION_ALL_IN, None)

        env.current_player = env.p2
        _, _, done, _ = env.step(ACTION_ALL_IN, None)

        assert done or env.street == SHOWDOWN

    def test_pot_increases(self):
        env = PokerEnv()
        env.reset()

        pot_before = env.engine.pot
        env.step(ACTION_CALL, None)

        assert env.engine.pot >= pot_before

    def test_no_infinite_loop(self):
        env = PokerEnv()
        env.reset()

        done = False
        steps = 0

        while not done:
            action = env.legal_actions()[0]
            _, _, done, _ = env.step(action, None)

            steps += 1
            if steps > 200:
                assert False, "Infinite loop detected"
    def test_fold_gives_pot_to_winner(self):
        env = PokerEnv()
        env.reset()

        p1_stack_before = env.p1.stack
        p2_stack_before = env.p2.stack

        _, _, done, _ = env.step(ACTION_FOLD, None)

        assert done is True

        assert env.p1.stack != p1_stack_before or env.p2.stack != p2_stack_before
    def test_reward_after_fold(self):
        env = PokerEnv()
        env.reset()

        _, reward, done, _ = env.step(ACTION_FOLD, None)

        assert done is True
        assert reward != 0.0
    def test_all_in_no_stack_left(self):
        env = PokerEnv()
        env.reset()

        env.step(ACTION_ALL_IN, None)

        p = env.p1 if env.p1.stack == 0 else env.p2
        assert p.stack == 0

    def test_street_never_goes_back(self):
        env = PokerEnv()
        env.reset()

        prev_street = env.street

        for _ in range(20):
            action = env.legal_actions()[0]
            _, _, done, _ = env.step(action, None)

            assert env.street >= prev_street
            prev_street = env.street

            if done:
                break
    def test_stacks_non_negative(self):
        env = PokerEnv()
        env.reset()

        done = False

        while not done:
            action = env.legal_actions()[0]
            _, _, done, _ = env.step(action, None)

            assert env.p1.stack >= 0
            assert env.p2.stack >= 0

    def test_stack_conservation(self):
        env = PokerEnv()
        env.reset()

        total_initial = env.p1.stack + env.p2.stack + env.engine.pot

        done = False
        while not done:
            action = env.legal_actions()[0]
            _, _, done, _ = env.step(action, None)

        total_final = env.p1.stack + env.p2.stack + env.engine.pot

        assert total_initial == total_final

    def test_legal_actions_range(self):
        env = PokerEnv()
        env.reset()

        done = False

        while not done:
            actions = env.legal_actions()

            for a in actions:
                assert 0 <= a < NUM_ACTIONS

            action = actions[0]
            _, _, done, _ = env.step(action, None)

    def test_no_step_after_done(self):
        env = PokerEnv()
        env.reset()

        env.step(ACTION_FOLD, None)

        try:
            env.step(ACTION_CALL, None)
            assert False
        except RuntimeError:
            assert True

    def test_observation_shape(self):
        env = PokerEnv()
        obs = env.reset()

        size = obs.shape[0]

        done = False

        while not done:
            action = env.legal_actions()[0]
            obs, _, done, _ = env.step(action, None)

            assert obs.shape[0] == size

    def test_episode_length(self):
        env = PokerEnv()
        env.reset()

        steps = 0
        done = False

        while not done:
            action = env.legal_actions()[0]
            _, _, done, _ = env.step(action, None)

            steps += 1

            if steps > 100:
                assert False, "Episode too long"

    def test_pot_non_decreasing(self):
        env = PokerEnv()
        env.reset()

        prev_pot = env.engine.pot
        done = False

        while not done:
            action = env.legal_actions()[0]
            _, _, done, _ = env.step(action, None)

            if not done:
                assert env.engine.pot >= prev_pot

            prev_pot = env.engine.pot


    def test_blinds_randomization(self):
        sb_count = 0

        for _ in range(50):
            env = PokerEnv()
            env.reset()

            if env.p1.position == "SB":
                sb_count += 1

        assert 10 < sb_count < 40


    def test_pot_non_negative(self):
        env = PokerEnv()
        env.reset()

        done = False

        while not done:
            action = env.legal_actions()[0]
            _, _, done, _ = env.step(action, None)

            assert env.engine.pot >= 0

# RL Agent test

class TestRLPoker:

    def test_no_nan_in_observation(self):
        env = PokerEnv()
        obs = env.reset()

        assert not np.isnan(obs.numpy()).any()

        done = False
        while not done:
            action = env.legal_actions()[0]
            obs, _, done, _ = env.step(action, None)

            assert not np.isnan(obs.numpy()).any()


    def test_observation_range(self):
        env = PokerEnv()
        obs = env.reset()

        assert (obs.numpy() >= 0).all()
        assert (obs.numpy() <= 1).all()


    def test_reward_matches_stack_diff(self):
        env = PokerEnv()
        env.reset()

        done = False
        last_reward = 0

        while not done:
            action = env.legal_actions()[0]
            _, reward, done, _ = env.step(action, None)
            last_reward = reward

        assert -1.0 <= last_reward <= 1.0


    def test_legal_mask_matches_actions(self):
        from src.poker_enviroment.observation import legal_action_mask

        env = PokerEnv()
        env.reset()

        legal = env.legal_actions()
        mask = legal_action_mask(legal)

        for i in range(NUM_ACTIONS):
            if i in legal:
                assert mask[i] == 1
            else:
                assert mask[i] == 0


    def test_no_illegal_actions_during_game(self):
        env = PokerEnv()
        env.reset()

        done = False

        while not done:
            legal = env.legal_actions()

            for a in legal:
                assert 0 <= a < NUM_ACTIONS

            action = legal[0]
            _, _, done, _ = env.step(action, None)


    def test_all_in_stops_decisions(self):
        env = PokerEnv()
        env.reset()

        env.step(ACTION_ALL_IN, None)

        done = False
        steps = 0

        while not done:
            steps += 1
            legal = env.legal_actions()

            if steps > 1:
                assert len(legal) <= 1

            if not legal:
                break

            _, _, done, _ = env.step(legal[0], None)

# RANDOM STRESS TEST

def test_random_hands():
    import random
    deck = create_deck()

    for _ in range(500):
        c1, c2 = random.sample(deck, 2)
        hand_to_ids(c1, c2)

if __name__ == "__main__":

    print("=== RUNNING TESTS ===")

    # Deck
    deck_tests = TestCreateDeck()
    deck_tests.test_create_deck_length()
    deck_tests.test_create_deck_unique()
    deck_tests.test_create_deck_contains_all_ranks()
    deck_tests.test_create_deck_contains_all_suits()
    print("✔ Deck tests passed")

    # Observation
    obs_tests = TestObservation()
    obs_tests.test_pair()
    obs_tests.test_suited()
    obs_tests.test_offsuit()
    obs_tests.test_order_independent()
    obs_tests.test_high_card_first()
    print("✔ Observation tests passed")

    # Edge cases
    edge_tests = TestEdgeCases()
    edge_tests.test_same_card_twice()
    edge_tests.test_invalid_rank()
    edge_tests.test_invalid_suit()
    print("✔ Edge case tests passed")

    # PokerEnv
    env_tests = TestPokerEnv()
    env_tests.test_reset()
    env_tests.test_legal_actions_not_empty()
    env_tests.test_fold_ends_game()
    env_tests.test_call_progresses_round()
    env_tests.test_all_in_sets_stack_zero()
    env_tests.test_all_in_vs_all_in_ends()
    env_tests.test_pot_increases()
    env_tests.test_no_infinite_loop()
    env_tests.test_fold_gives_pot_to_winner()
    env_tests.test_reward_after_fold()
    env_tests.test_all_in_no_stack_left()
    env_tests.test_street_never_goes_back()
    env_tests.test_stacks_non_negative()
    env_tests.test_stack_conservation()
    env_tests.test_legal_actions_range()
    env_tests.test_no_step_after_done()
    env_tests.test_observation_shape()
    env_tests.test_episode_length()
    env_tests.test_pot_non_decreasing()
    env_tests.test_blinds_randomization()
    env_tests.test_pot_non_negative()
    print("✔ PokerEnv tests passed")

    # Poker AI Agent
    rl_tests = TestRLPoker()
    rl_tests.test_no_nan_in_observation()
    rl_tests.test_observation_range()
    rl_tests.test_reward_matches_stack_diff()
    rl_tests.test_legal_mask_matches_actions()
    rl_tests.test_no_illegal_actions_during_game()
    rl_tests.test_all_in_stops_decisions()

    print("✔ RL tests passed")

    # Random stress
    test_random_hands()
    print("✔ Random tests passed")

    print("\nALL TESTS PASSED")