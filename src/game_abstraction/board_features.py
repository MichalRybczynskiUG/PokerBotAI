#!/usr/bin/env python
# coding: utf-8

# In[2]:

import numpy as np


# In[ ]:


# FEATURE VECTOR FOR A FLOP

def extract_flop_board_features(card1, card2, card3):
    cards = [card1, card2, card3]

    # --- ranks and suits ---
    ranks = sorted(
        (c.rank for c in cards),
        reverse=True
    )
    suits = [c.suit for c in cards]

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
        high,
        mid,
        low,
        average_rank,
        broadway_count,
        has_ace,
        paired_board_strength,
        gap_high_mid,
        gap_mid_low,
        connectivity_quality,
        straight_draw_potential,
        nut_straight_possible,
        wheel_possible,
        flush_draw_potential,
        ace_suited_on_board,
        nut_flush_possible,
        broadway_connectivity,
        is_dynamic,
        nut_density,
    ], dtype=float)


# In[ ]:


# FEATURE VECTOR FOR A TURN

def extract_turn_board_features(card1, card2, card3, card4):
    """
    Extracts a numerical feature vector describing turn board texture.

    The features encode rank structure, pairing and multiplicities,
    connectivity, straight and flush completion or draw potential,
    broadway presence, board dynamics, and proxy measures of nut density.
    The representation is intended for statistical analysis or machine
    learning models operating on turn boards.

    Parameters
    ----------
    card1, card2, card3, card4 : Card
        Turn board cards.

    Returns
    -------
    np.ndarray
        1D array of float features representing the turn board texture.
    """
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

