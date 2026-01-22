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
import pickle
import time
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from treys import Card, Evaluator


# %%
# 1. FEATURE VECTOR FOR A FLOP

def flop_features_solver(card1, card2, card3):
    cards = [card1, card2, card3]

    # --- ranks and suits ---
    ranks = sorted(
        (Card.get_rank_int(c) for c in cards),
        reverse=True
    )
    suits = [Card.get_suit_int(c) for c in cards]

    high, mid, low = ranks

    # --- basic rank structure ---
    broadway_count = sum(r >= 8 for r in ranks)   # T,J,Q,K,A
    has_ace = int(high == 12)
    average_rank = (high + mid + low) / 3.0

    is_paired = int(high == mid or mid == low)
    is_trips = int(high == mid == low)

    paired_board_strength = (
        2 if is_trips else
        1 if is_paired else
        0
    )

    # --- gaps / connectivity ---
    gap_high_mid = high - mid
    gap_mid_low = mid - low
    max_gap = max(gap_high_mid, gap_mid_low)

    # connectivity (0–3)
    # 3 = very connected (np. JT9)
    # 2 = semi-connected (np. QJ8)
    # 1 = loose (np. KJ4)
    # 0 = no connected
    connectivity_quality = (
        3 if max_gap == 1 else
        2 if max_gap == 2 else
        1 if max_gap == 3 else
        0
    )

    # --- straight potential ---
    straight_draw_potential = max(
        0,
        (4 - gap_high_mid) + (4 - gap_mid_low)
    )

    nut_straight_possible = int(
        gap_high_mid == 1 and gap_mid_low == 1 and high >= 9
    )

    # wheel A2345
    wheel_possible = int(
        has_ace and mid <= 3
    )

    # --- flush potential ---
    suit_count = len(set(suits))
    flush_draw_potential = 3 - suit_count

    ace_suited_on_board = int(
        has_ace and flush_draw_potential >= 1
    )

    nut_flush_possible = ace_suited_on_board

    # --- broadway texture ---
    broadway_connectivity = (
        2 if broadway_count >= 2 and high - low <= 4 else
        1 if broadway_count >= 2 else
        0
    )

    # --- board dynamics ---
    is_dynamic = int(
        straight_draw_potential +
        flush_draw_potential +
        connectivity_quality >= 6
    )

    # --- nut density proxy ---
    nut_density = (
        broadway_connectivity +
        straight_draw_potential +
        flush_draw_potential +
        nut_straight_possible +
        nut_flush_possible
    )

    return np.array([
        # rank structure
        high,
        mid,
        low,
        average_rank,
        broadway_count,
        has_ace,

        # pairing
        paired_board_strength,

        # connectivity
        gap_high_mid,
        gap_mid_low,
        connectivity_quality,

        # straights
        straight_draw_potential,
        nut_straight_possible,
        wheel_possible,

        # flushes
        flush_draw_potential,
        ace_suited_on_board,
        nut_flush_possible,

        # broadway texture
        broadway_connectivity,

        # dynamics / nuts
        is_dynamic,
        nut_density,
    ], dtype=float)


# %%
# 2. ALL FLOPS

def all_flop_vectors_solver():
    deck = [Card.new(r+s) for r in "23456789TJQKA" for s in "shdc"]
    vecs, flops = [], []

    for c1, c2, c3 in combinations(deck, 3):
        vecs.append(flop_features_solver(c1, c2, c3))
        flops.append((c1, c2, c3))

    return np.array(vecs), flops



# %%
# =========================================================
# 1. HAND CLASSES
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
    deck = [Card.new(r+s) for r in RANK_CHARS for s in "shdc"]
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
# 2. SINGLE RUNOUT SIMULATION
# =========================================================

def simulate_runout(hero_hand, flop):
    used = set(flop) | set(hero_hand)
    remaining = [c for c in DECK_ALL if c not in used]

    opp_hand = random.sample(remaining, 2)
    used2 = used | set(opp_hand)

    remaining2 = [c for c in DECK_ALL if c not in used2]
    turn, river = random.sample(remaining2, 2)

    board = list(flop) + [turn, river]

    hero_score = evaluator.evaluate(board, list(hero_hand))
    opp_score = evaluator.evaluate(board, opp_hand)

    if hero_score < opp_score:
        return 1.0
    elif hero_score == opp_score:
        return 0.5
    else:
        return 0.0

# =========================================================
# 3. HAND-LEVEL METRICS (MINIMAL SET)
# =========================================================

def hand_metrics_on_flop(hero_hand, flop, samples):
    results = np.array([simulate_runout(hero_hand, flop) for _ in range(samples)])

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
    flops_per_bucket=75,
    turn_river_samples=150,
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

    sampled_flops = random.sample(
        flops_in_bucket,
        min(len(flops_in_bucket), flops_per_bucket)
    )

    # =====================================================
    # MAIN LOOP WITH PROGRESS BAR (ETA)
    # =====================================================

    for flop in tqdm(
        sampled_flops,
        desc="SIMULATING FLOPS",
        unit="flop"
    ):
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
                        turn_river_samples
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

    print(f"\nCzas simulate_bucket_metrics: {time.time() - start:.2f} s")

    return final, range_metrics



# %%
def save_flop_buckets_only(flops, labels, out_file="flop_buckets.pkl"):
    """
    Save only flop buckets (without any metrics).
    
    buckets[bucket_id] = [(c1, c2, c3), ...]
    """
    buckets = defaultdict(list)

    for flop, label in zip(flops, labels):
        buckets[label].append(flop)

    buckets = dict(buckets)

    with open(out_file, "wb") as f:
        pickle.dump(buckets, f)

    print(f"Saved {len(buckets)} flop buckets to {out_file}")
    return buckets


# %%
def compute_and_save_all_bucket_metrics(
    flops,
    labels,
    num_buckets,
    flops_per_bucket=15,
    turn_river_samples=15,
    out_file="bucket_flop_metrics.pkl",
    seed=42
):
    all_results = {}

    for bucket_id in range(num_buckets):
        print(f"Bucket {bucket_id}")

        flops_in_bucket = [
            flops[i] for i, b in enumerate(labels) if b == bucket_id
        ]

        if not flops_in_bucket:
            print("  empty, skipping")
            continue

        bucket_metrics, range_metrics = simulate_bucket_metrics(
            flops_in_bucket,
            flops_per_bucket=flops_per_bucket,
            turn_river_samples=turn_river_samples,
            seed=seed + bucket_id
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



# %%
if __name__ == "__main__":

    FLOPS_PER_BUCKET = 50
    TURN_RIVER_SAMPLES = 200
    
    vecs, flops = all_flop_vectors_solver()

    K = 25
    vecs_norm = (vecs - vecs.mean(axis=0)) / vecs.std(axis=0)
    kmeans = KMeans(n_clusters=K, n_init='auto').fit(vecs_norm)

    labels = kmeans.labels_

    RANK_CHARS = "23456789TJQKA"

    class_to_hands, ordered_hand_classes = build_hand_classes()

    evaluator = Evaluator()
    DECK_ALL = [Card.new(r+s) for r in RANK_CHARS for s in "shdc"]

    flop_buckets = save_flop_buckets_only(
        flops=flops,
        labels=labels,
        out_file="flop_buckets_v2.pkl")
    
    all_bucket_results = compute_and_save_all_bucket_metrics(
    flops=flops,
    labels=labels,
    num_buckets=K,
    flops_per_bucket=FLOPS_PER_BUCKET,
    turn_river_samples=TURN_RIVER_SAMPLES,
    out_file="flop_bucket_metrics_v2.pkl"
)
