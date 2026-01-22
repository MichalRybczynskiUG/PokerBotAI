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
import random
import time
from tqdm import tqdm
import pickle
from itertools import combinations
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from treys import Card, Evaluator


# %%
def turn_features_solver(card1, card2, card3, card4):
    cards = [card1, card2, card3, card4]

    # --- ranks and suits ---
    ranks = sorted(
        (Card.get_rank_int(c) for c in cards),
        reverse=True
    )
    suits = [Card.get_suit_int(c) for c in cards]

    high, mid1, mid2, low = ranks

    # --- rank statistics ---
    average_rank = sum(ranks) / 4.0
    broadway_count = sum(r >= 8 for r in ranks)   # T,J,Q,K,A
    has_ace = int(high == 12)

    # --- multiplicities ---
    rank_counts = {r: ranks.count(r) for r in set(ranks)}
    counts_sorted = sorted(rank_counts.values(), reverse=True)

    is_quads = int(4 in counts_sorted)
    is_trips = int(3 in counts_sorted)
    is_paired = int(2 in counts_sorted)
    is_two_pair = int(counts_sorted.count(2) == 2)

    paired_board_strength = (
        3 if is_quads else
        2 if is_trips else
        1 if is_two_pair or is_paired else
        0
    )

    # --- gaps / connectivity ---
    gaps = [ranks[i] - ranks[i+1] for i in range(3)]
    max_gap = max(gaps)

    connectivity_quality = (
        3 if max_gap <= 1 else
        2 if max_gap == 2 else
        1 if max_gap == 3 else
        0
    )

    # --- straight analysis ---
    unique_ranks = sorted(set(ranks), reverse=True)

    straight_made = int(
        len(unique_ranks) == 4 and
        unique_ranks[0] - unique_ranks[3] == 3
    )

    straight_draw_potential = max(
        0,
        sum(4 - g for g in gaps)
    )

    nut_straight_possible = int(
        straight_made and high >= 9
    )

    # --- flush analysis ---
    suit_counts = {s: suits.count(s) for s in set(suits)}
    max_suit = max(suit_counts.values())

    flush_made = int(max_suit == 4)
    flush_draw = int(max_suit == 3)

    ace_suited_on_board = int(
        has_ace and max_suit >= 3
    )

    nut_flush_possible = int(flush_made and ace_suited_on_board)

    # --- broadway texture ---
    broadway_connectivity = (
        2 if broadway_count >= 3 and high - low <= 4 else
        1 if broadway_count >= 2 else
        0
    )

    # --- board dynamics ---
    is_dynamic = int(
        straight_draw_potential +
        flush_draw +
        connectivity_quality >= 7
    )

    # --- nut density proxy ---
    nut_density = (
        broadway_connectivity +
        straight_made +
        flush_made +
        nut_straight_possible +
        nut_flush_possible
    )

    return np.array([
        # rank structure
        high,
        mid1,
        mid2,
        low,
        average_rank,
        broadway_count,
        has_ace,

        # pairing
        paired_board_strength,
        is_two_pair,
        is_trips,
        is_quads,

        # connectivity
        max_gap,
        connectivity_quality,

        # straights
        straight_made,
        straight_draw_potential,
        nut_straight_possible,

        # flushes
        flush_made,
        flush_draw,
        ace_suited_on_board,
        nut_flush_possible,

        # broadway texture
        broadway_connectivity,

        # dynamics / nuts
        is_dynamic,
        nut_density,
    ], dtype=float)



# %%
def all_turn_vectors_solver():
    deck = [Card.new(r + s) for r in "23456789TJQKA" for s in "shdc"]
    vecs, turns = [], []

    for c1, c2, c3, c4 in combinations(deck, 4):
        vecs.append(
            turn_features_solver(c1, c2, c3, c4)
        )
        turns.append((c1, c2, c3, c4))

    return np.array(vecs), turns


# %%
# =========================================================
# 1. HAND CLASSES (169)
# =========================================================

def rank_int_to_char(r):
    return RANK_CHARS[r]

def hand_class_key(card1, card2):
    r1, r2 = Card.get_rank_int(card1), Card.get_rank_int(card2)
    s1, s2 = Card.get_suit_int(card1), Card.get_suit_int(card2)

    c1, c2 = rank_int_to_char(r1), rank_int_to_char(r2)

    if r1 == r2:
        return c1 + c2

    if r1 < r2:
        r1, r2 = r2, r1
        c1, c2 = c2, c1
        s1, s2 = s2, s1

    return c1 + c2 + ('s' if s1 == s2 else 'o')


def build_hand_classes():
    deck = [Card.new(r + s) for r in RANK_CHARS for s in "shdc"]
    class_to_hands = defaultdict(list)

    for c1, c2 in combinations(deck, 2):
        key = hand_class_key(c1, c2)
        class_to_hands[key].append((c1, c2))

    ordered = []
    for hi in reversed(RANK_CHARS):
        for lo in reversed(RANK_CHARS):
            if hi == lo:
                ordered.append(hi + lo)
            elif RANK_CHARS.index(hi) > RANK_CHARS.index(lo):
                ordered.append(hi + lo + 's')
                ordered.append(hi + lo + 'o')

    return class_to_hands, ordered

# =========================================================
# 3. SINGLE TURN RUNOUT (ONLY RIVER)
# =========================================================

def simulate_turn_runout(hero_hand, board4):
    used = set(board4) | set(hero_hand)
    remaining = [c for c in DECK_ALL if c not in used]

    opp_hand = random.sample(remaining, 2)
    used |= set(opp_hand)

    river = random.choice([c for c in remaining if c not in opp_hand])
    board = list(board4) + [river]

    hero_score = evaluator.evaluate(board, list(hero_hand))
    opp_score = evaluator.evaluate(board, opp_hand)

    if hero_score < opp_score:
        return 1.0
    elif hero_score == opp_score:
        return 0.5
    else:
        return 0.0


# =========================================================
# 4. HAND METRICS ON TURN
# =========================================================

def hand_metrics_on_turn(hero_hand, board4, samples):
    results = np.array([
        simulate_turn_runout(hero_hand, board4)
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
# 5. TURN BUCKET-LEVEL METRICS
# =========================================================

def simulate_turn_bucket_metrics(
    turns_in_bucket,
    turns_per_bucket=20,
    river_samples=30,
    seed=42
):
    start = time.time()

    random.seed(seed)
    np.random.seed(seed)

    bucket = {
        k: {
            "EHS": [], "VAR": [], "MED": [], "IQR": [],
            "NEG_POT": [], "POS_POT": [],
            "CRASH": [], "DOM": []
        }
        for k in ordered_hand_classes
    }

    sampled_turns = random.sample(
        turns_in_bucket,
        min(len(turns_in_bucket), turns_per_bucket)
    )

    # =====================================================
    # MAIN LOOP WITH PROGRESS BAR (ETA)
    # =====================================================

    for board4 in tqdm(
        sampled_turns,
        desc="SIMULATING TURNS",
        unit="turn"
    ):
        used_board = set(board4)

        for hand_key in ordered_hand_classes:
            vals = []

            for hero_hand in class_to_hands[hand_key]:
                if hero_hand[0] in used_board or hero_hand[1] in used_board:
                    continue

                vals.append(
                    hand_metrics_on_turn(
                        hero_hand,
                        board4,
                        river_samples
                    )
                )

            if vals:
                for k in bucket[hand_key]:
                    bucket[hand_key][k].append(
                        np.mean([v[k] for v in vals])
                    )

    # =====================================================
    # FINAL AGGREGATION
    # =====================================================

    final = {
        k: {kk: np.nanmean(vv) for kk, vv in v.items()}
        for k, v in bucket.items()
    }

    ehs_vals = np.array([
        v["EHS"] for v in final.values()
        if not np.isnan(v["EHS"])
    ])

    var_vals = np.array([
        v["VAR"] for v in final.values()
        if not np.isnan(v["VAR"])
    ])

    range_metrics = {
        "RangeAdv": ehs_vals.mean() - 0.5,
        "NutAdv": np.mean(ehs_vals > 0.8),
        "EPI": np.mean(ehs_vals < 0.2) + np.mean(ehs_vals > 0.8),
        "ECI": np.std(ehs_vals),
        "ShowdownDensity": np.mean((ehs_vals > 0.45) & (ehs_vals < 0.65)),
        "LockIn": 1.0 - np.mean(var_vals)
    }

    print(f"\nCzas simulate_turn_bucket_metrics: {time.time() - start:.2f} s")

    return final, range_metrics



# %%
def save_turn_buckets_only(turns, labels, out_file="turn_buckets.pkl"):
    """
    Save only turn buckets (without any metrics).

    buckets[bucket_id] = [(c1, c2, c3, c4), ...]
    """
    buckets = defaultdict(list)

    for turn, label in zip(turns, labels):
        buckets[label].append(turn)

    buckets = dict(buckets)

    with open(out_file, "wb") as f:
        pickle.dump(buckets, f)

    print(f"Saved {len(buckets)} turn buckets to {out_file}")
    return buckets



# %%
def compute_and_save_all_turn_bucket_metrics(
    turns,
    labels,
    num_buckets,
    turns_per_bucket=20,
    river_samples=30,
    out_file="bucket_turn_metrics.pkl",
    seed=42
):
    all_results = {}

    for bucket_id in range(num_buckets):
        print(f"Turn Bucket {bucket_id}")

        turns_in_bucket = [
            turns[i] for i, b in enumerate(labels) if b == bucket_id
        ]

        if not turns_in_bucket:
            print("  empty, skipping")
            continue

        bucket_metrics, range_metrics = simulate_turn_bucket_metrics(
            turns_in_bucket,
            turns_per_bucket=turns_per_bucket,
            river_samples=river_samples,
            seed=seed + bucket_id
        )

        all_results[bucket_id] = {
            "hand_metrics": bucket_metrics,
            "range_metrics": range_metrics,
            "num_turns": len(turns_in_bucket)
        }

    with open(out_file, "wb") as f:
        pickle.dump(all_results, f)

    print(f"\nSaved TURN bucket metrics to {out_file}")
    return all_results



# %%
if __name__ == "__main__":

    TURNS_PER_BUCKET = 100
    RIVER_SAMPLES = 300

    turn_vecs, turns = all_turn_vectors_solver()
    
    K = 25 
    
    turn_vecs_norm = (
        turn_vecs - turn_vecs.mean(axis=0)
    ) / turn_vecs.std(axis=0)
    
    kmeans_turn = KMeans(
        n_clusters=K,
        n_init="auto",
        random_state=0
    ).fit(turn_vecs_norm)
    
    turn_labels = kmeans_turn.labels_

    RANK_CHARS = "23456789TJQKA"

    class_to_hands, ordered_hand_classes = build_hand_classes()
    
    evaluator = Evaluator()
    DECK_ALL = [Card.new(r + s) for r in RANK_CHARS for s in "shdc"]

    turn_buckets = save_turn_buckets_only(
        turns=turns,
        labels=turn_labels,
        out_file="turn_buckets.pkl"
    )

    num_buckets = len(set(turn_labels))

    turn_bucket_metrics = compute_and_save_all_turn_bucket_metrics(
        turns=turns,
        labels=turn_labels,
        num_buckets=num_buckets,
        turns_per_bucket=TURNS_PER_BUCKET,
        river_samples=RIVER_SAMPLES,
        out_file="turn_bucket_metrics_v2.pkl"
    )
