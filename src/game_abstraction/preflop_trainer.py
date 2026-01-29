# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import itertools
import random
import time
import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from treys import Card, Deck, Evaluator


# %%
def generate_pocket_ids():

    """
    Generate all 169 distinct starting hand identifiers (pocket IDs) in Texas Hold'em.
    
    Each pocket ID is represented as a tuple (r1, r2, suited):
        - r1 <= r2 (canonical ordering)
        - suited is False for pairs

        - suited : boolean flag
                   True  -> both cards share the same suit (suited)
                   False -> cards have different suits (offsuit)
    Returns
    -------
    list of tuples
        A list of 169 pocket descriptors, where each element is (r1, r2, suited).
    """
    
    ranks = list(range(13))  # 0=2, 12=A
    pockets = []
    for r1, r2 in itertools.combinations_with_replacement(ranks, 2):
        if r1 == r2:
            pockets.append((r1, r2, False))
        else:
            pockets.append((r1, r2, True))
            pockets.append((r1, r2, False))
    return pockets

def random_pocket_from_id(pid):
    """
      Generate a random concrete hand (two actual cards) corresponding to a given pocket ID.

    Parameters
    ----------
    pid : tuple
        A pocket descriptor of the form (r1, r2, suited)

    Returns
    -------
    list of str
        A list containing two card strings in the format "RS", where:
            - R is the rank symbol
            - S is the suit
    """
    
    r1, r2, suited = pid
    ranks = "23456789TJQKA"
    suits = "shdc"

    if r1 == r2: #pair
        s1, s2 = random.sample(suits, 2)
        return [ranks[r1] + s1, ranks[r2] + s2]

    c1 = ranks[r1] + random.choice(suits)
    if suited:
        c2 = ranks[r2] + c1[1]
    else:
        c2 = ranks[r2] + random.choice([s for s in suits if s != c1[1]])

    return [c1, c2]

def hand_score(hero_cards, board_cards):
    """
    Evaluate rank
    """
    hero = [Card.new(c) for c in hero_cards]
    board = [Card.new(c) for c in board_cards]
    return evaluator.evaluate(hero, board)


# =========================================================
# SINGLE PREFLOP RUNOUT
# =========================================================

def simulate_preflop_runout(hero):
    deck = Deck()
    deck.cards.remove(hero[0])
    deck.cards.remove(hero[1])

    opp = random.sample(deck.cards, 2)
    for c in opp:
        deck.cards.remove(c)

    board = random.sample(deck.cards, 5)

    hero_rank = evaluator.evaluate(hero, board)
    opp_rank  = evaluator.evaluate(opp, board)

    if hero_rank < opp_rank:
        return 1.0
    elif hero_rank == opp_rank:
        return 0.5
    else:
        return 0.0


# =========================================================
# PREFLOP METRICS FOR ONE POCKET
# =========================================================

def compute_preflop_metrics(hero_cards, samples=5000):
    hero = [Card.new(c) for c in hero_cards]

    results = np.array([
        simulate_preflop_runout(hero)
        for _ in range(samples)
    ])

    return {
        "EHS": results.mean(),
        "VAR": results.var(),
        "MED": np.median(results),
        "IQR": np.quantile(results, 0.75) - np.quantile(results, 0.25),
        "NEG_POT": np.mean(results < 0.5),
        "POS_POT": np.mean(results > 0.5)
    }


# =========================================================
# FULL 169-HAND PREFLOP TABLE
# =========================================================

def build_preflop_metrics_table(samples_per_pocket=5000):
    pockets = generate_pocket_ids()
    results = {}

    start = time.time()

    for pid in tqdm(pockets, desc="PREFLOP METRICS"):
        hero = random_pocket_from_id(pid)
        results[pid] = compute_preflop_metrics(
            hero,
            samples=samples_per_pocket
        )

    print("\nCzas budowy PREFLOP:", round(time.time() - start, 2), "s")
    return results

def save_ehs_pickle(ehs, filename="preflop_metrics.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(ehs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved in:", filename)


# %%
if __name__ == "__main__":
    evaluator = Evaluator()
    ehs = build_preflop_metrics_table(samples_per_pocket=100000)
    save_ehs_pickle(ehs)
