#!/usr/bin/env python
# coding: utf-8

# In[4]:

import random
import time
import pickle
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm

from sklearn.cluster import KMeans

from treys import Card, Evaluator

from board_features import extract_flop_board_features


# In[4]:


def all_flop_feature_extractors():
    deck = [Card.new(r+s) for r in "23456789TJQKA" for s in "shdc"]
    vecs, flops = [], []

    for c1, c2, c3 in combinations(deck, 3):
        vecs.append(extract_flop_board_features(c1, c2, c3))
        flops.append((c1, c2, c3))

    return np.array(vecs), flops


# In[5]:


# =========================================================
# 1. HAND CLASSES
# =========================================================

def rank_int_to_char(r, rank_chars):
    return rank_chars[r]


def hand_class_key(card1, card2, rank_chars):
    r1, r2 = Card.get_rank_int(card1), Card.get_rank_int(card2)
    s1, s2 = Card.get_suit_int(card1), Card.get_suit_int(card2)

    c1 = rank_int_to_char(r1, rank_chars)
    c2 = rank_int_to_char(r2, rank_chars)

    if r1 == r2:
        return c1 + c2

    if r1 < r2:
        r1, r2 = r2, r1
        c1, c2 = c2, c1
        s1, s2 = s2, s1

    return c1 + c2 + ('s' if s1 == s2 else 'o')


def build_hand_classes(rank_chars):
    deck = [Card.new(r + s) for r in rank_chars for s in "shdc"]
    class_to_hands = defaultdict(list)

    for c1, c2 in combinations(deck, 2):
        key = hand_class_key(c1, c2, rank_chars)
        class_to_hands[key].append((c1, c2))

    ordered = []
    for hi in reversed(rank_chars):
        for lo in reversed(rank_chars):
            if hi == lo:
                ordered.append(hi + lo)
            elif rank_chars.index(hi) > rank_chars.index(lo):
                ordered.append(hi + lo + 's')
                ordered.append(hi + lo + 'o')

    return class_to_hands, ordered


# =========================================================
# 2. SINGLE RUNOUT SIMULATION
# =========================================================

def simulate_runout(hero_hand, flop, deck_all, evaluator):
    used = set(flop) | set(hero_hand)
    remaining = [c for c in deck_all if c not in used]

    opp_hand = random.sample(remaining, 2)
    used2 = used | set(opp_hand)

    remaining2 = [c for c in deck_all if c not in used2]
    turn, river = random.sample(remaining2, 2)

    board = list(flop) + [turn, river]

    hero_score = evaluator.evaluate(board, list(hero_hand))
    opp_score  = evaluator.evaluate(board, opp_hand)

    if hero_score < opp_score:
        return 1.0
    elif hero_score == opp_score:
        return 0.5
    else:
        return 0.0


# =========================================================
# 3. HAND-LEVEL METRICS
# =========================================================

def hand_metrics_on_flop(hero_hand, flop, samples, deck_all, evaluator):
    results = np.array([
        simulate_runout(hero_hand, flop, deck_all, evaluator)
        for _ in range(samples)
    ])

    return {
        "EHS": results.mean(),
        "VAR": results.var(),
        "MED": np.median(results),
        "IQR": np.quantile(results, 0.75) - np.quantile(results, 0.25),
        "NEG_POT": np.mean(results < 0.5),
        "POS_POT": np.mean(results > 0.5),
        "CRASH": np.mean(results < 0.2),
        "DOM": np.mean(results > 0.8)
    }


# =========================================================
# 4. BUCKET-LEVEL METRICS
# =========================================================

def simulate_bucket_metrics(
    flops_in_bucket,
    class_to_hands,
    ordered_hand_classes,
    deck_all,
    evaluator,
    flops_per_bucket=25,
    turn_river_samples=150,
    seed=42
):
    start = time.time()
    random.seed(seed)
    np.random.seed(seed)

    bucket = {
        k: {m: [] for m in ["EHS","VAR","MED","IQR","NEG_POT","POS_POT","CRASH","DOM"]}
        for k in ordered_hand_classes
    }

    sampled_flops = random.sample(
        flops_in_bucket,
        min(len(flops_in_bucket), flops_per_bucket)
    )

    for flop in tqdm(sampled_flops, desc="SIMULATING FLOPS", unit="flop"):
        used_flop = set(flop)

        for hand_key in ordered_hand_classes:
            vals = []

            for hero_hand in class_to_hands[hand_key]:
                if hero_hand[0] in used_flop or hero_hand[1] in used_flop:
                    continue

                vals.append(
                    hand_metrics_on_flop(
                        hero_hand,
                        flop,
                        turn_river_samples,
                        deck_all,
                        evaluator
                    )
                )

            if vals:
                for m in bucket[hand_key]:
                    bucket[hand_key][m].append(
                        np.mean([v[m] for v in vals])
                    )

    final = {
        k: {m: np.nanmean(vs) for m, vs in v.items()}
        for k, v in bucket.items()
    }

    ehs_vals = np.array([v["EHS"] for v in final.values() if not np.isnan(v["EHS"])])
    var_vals = np.array([v["VAR"] for v in final.values() if not np.isnan(v["VAR"])])

    range_metrics = {
        "RangeAdv": ehs_vals.mean() - 0.5,
        "NutAdv": np.mean(ehs_vals > 0.8),
        "EPI": np.mean(ehs_vals < 0.2) + np.mean(ehs_vals > 0.8),
        "ECI": np.std(ehs_vals),
        "ShowdownDensity": np.mean((ehs_vals > 0.45) & (ehs_vals < 0.65)),
        "LockIn": 1.0 - np.mean(var_vals)
    }

    print(f"\nBucket simulated in {time.time() - start:.2f}s")
    return final, range_metrics


# In[ ]:


# =========================================================
# 5. FULL PIPELINE
# =========================================================

def compute_and_save_all_bucket_metrics(
    flops,
    labels,
    num_buckets,
    class_to_hands,
    ordered_hand_classes,
    deck_all,
    evaluator,
    flops_per_bucket,
    turn_river_samples,
    out_file,
    seed=42
):
    all_results = {}

    for bucket_id in range(num_buckets):
        print(f"\nBUCKET {bucket_id}")

        flops_in_bucket = [
            flops[i] for i, b in enumerate(labels) if b == bucket_id
        ]

        if not flops_in_bucket:
            print("  empty, skipping")
            continue

        bucket_metrics, range_metrics = simulate_bucket_metrics(
            flops_in_bucket,
            class_to_hands,
            ordered_hand_classes,
            deck_all,
            evaluator,
            flops_per_bucket,
            turn_river_samples,
            seed + bucket_id
        )

        all_results[bucket_id] = {
            "hand_metrics": bucket_metrics,
            "range_metrics": range_metrics,
            "num_flops": len(flops_in_bucket)
        }

    with open(out_file, "wb") as f:
        pickle.dump(all_results, f)

    print(f"\nSaved bucket metrics to {out_file}")
    return all_results



# In[ ]:


def save_flop_abstraction(kmeans, mean, std, path="flop_abstraction.pkl"):
    with open(path, "wb") as f:
        pickle.dump({
            "kmeans": kmeans,
            "mean": mean,
            "std": std
        }, f)


# In[ ]:

def main():
    DATA_PATH = Path.cwd().parents[1] / "data"
    RANK_CHARS = "23456789TJQKA"

    FLOPS_PER_BUCKET = 40
    TURN_RIVER_SAMPLES = 125
    K = 25

    vecs, flops = all_flop_feature_extractors()

    mean = vecs.mean(axis=0)
    std  = vecs.std(axis=0)
    std  = np.where(std == 0, 1.0, std)

    vecs_norm = (vecs - mean) / std
    kmeans = KMeans(n_clusters=K, n_init="auto").fit(vecs_norm)
    labels = kmeans.labels_

    save_flop_abstraction(kmeans, mean, std, path=DATA_PATH / "flop_abstraction.pkl")

    class_to_hands, ordered_hand_classes = build_hand_classes(RANK_CHARS)
    evaluator = Evaluator()
    deck_all = [Card.new(r+s) for r in RANK_CHARS for s in "shdc"]

    compute_and_save_all_bucket_metrics(
        flops,
        labels,
        K,
        class_to_hands,
        ordered_hand_classes,
        deck_all,
        evaluator,
        FLOPS_PER_BUCKET,
        TURN_RIVER_SAMPLES,
        DATA_PATH / "flop_bucket_metrics.pkl"
    )


if __name__ == "__main__":
    main()