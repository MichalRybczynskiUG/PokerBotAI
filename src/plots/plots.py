#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path


# In[2]:


def generate_preflop_metric_heatmap(
    data,
    metric,
    title,
    filename,
    vmin=None,
    vmax=None
):
    ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
    size = 13

    matrix = np.zeros((size, size))
    counts = np.zeros((size, size))

    for (r1, r2, suited), metrics in data.items():
        value = metrics[metric]

        if r1 == r2:          # pairs → diagonal
            i, j = r1, r2
        elif suited:          # suited → lower triangle
            i, j = r2, r1
        else:                 # offsuit → upper triangle
            i, j = r1, r2

        matrix[i, j] += value
        counts[i, j] += 1

    matrix = np.divide(matrix, counts, out=np.zeros_like(matrix), where=counts > 0)

    # ---- colormap ----
    colors_main = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#64B5CD"]
    segment_size = 40
    full_colors = []

    for col in colors_main:
        base = np.array(mcolors.to_rgb(col))
        dark = base * 0.6
        light = np.clip(base * 1.3, 0, 1)
        segment = [
            tuple(dark + (light - dark) * (k / (segment_size - 1)))
            for k in range(segment_size)
        ]
        full_colors.extend(segment)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "segmented", full_colors, N=len(full_colors)
    )

    plt.figure(figsize=(8, 8))
    sns.heatmap(
        matrix,
        xticklabels=ranks,
        yticklabels=ranks,
        cmap=cmap,
        square=True,
        vmin=vmin,
        vmax=vmax,
        cbar=True
    )

    plt.title(title)
    plt.xlabel("Card 1")
    plt.ylabel("Card 2")

    for i in range(size):
        for j in range(size):
            if counts[i, j] > 0:
                plt.text(
                    j + 0.5, i + 0.5,
                    f"{matrix[i, j]:.2f}",
                    ha='center', va='center',
                    color='white', fontsize=7
                )

    plt.savefig(PLOT_PATH / filename, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()

    print(f"Zapisano wykres: {filename}")

def gen_preflop_plots(filename = "preflop_ehs.pkl"):
    PRE_FLOP_METRICS = {
        "EHS":     (0.0, 1.0),
        "VAR":     (0.0, None),
        "MED":     (0.0, 1.0),
        "IQR":     (0.0, None),
        "NEG_POT": (0.0, 1.0),
        "POS_POT": (0.0, 1.0),
    }

    with open(DATA_PATH /filename, "rb") as f:
        pre = pickle.load(f)

    for metric, (vmin, vmax) in PRE_FLOP_METRICS.items():
        generate_preflop_metric_heatmap(
            data=pre,
            metric=metric,
            title=f"Preflop {metric} Heatmap",
            filename=f"preflop_{metric.lower()}.png",
            vmin=vmin,
            vmax=vmax
        )


# In[3]:


def hand_key_to_tuple(hand_key):

    if len(hand_key) == 2:
        r = RANK_TO_INDEX[hand_key[0]]
        return (r, r, False)

    r1 = RANK_TO_INDEX[hand_key[0]]
    r2 = RANK_TO_INDEX[hand_key[1]]
    suited = (hand_key[2] == 's')
    return (r1, r2, suited)


def plot_bucket_metric_heatmap(
    bucket_results,
    bucket_id,
    metric,
    filename,
    vmin=None,
    vmax=None
):
    """
    Plot a 13x13 hand-class heatmap for a given flop bucket and metric.
    """

    size = 13
    ranks = list(RANKS)

    matrix = np.zeros((size, size))
    counts = np.zeros((size, size))

    hand_metrics = bucket_results[bucket_id]["hand_metrics"]

    # ---- fill matrix ----
    for hand_key, metrics in hand_metrics.items():
        value = metrics.get(metric)
        if value is None or np.isnan(value):
            continue

        r1, r2, suited = hand_key_to_tuple(hand_key)

        if r1 == r2:          # pairs → diagonal
            i, j = r1, r2
        elif suited:          # suited → lower triangle
            i, j = r2, r1
        else:                 # offsuit → upper triangle
            i, j = r1, r2

        matrix[i, j] += value
        counts[i, j] += 1

    matrix = np.divide(matrix, counts, out=np.zeros_like(matrix), where=counts > 0)

    # ---- colormap (same style as preflop) ----
    base_colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#64B5CD"]
    labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]

    full_colors = []
    for col in base_colors:
        base = np.array(mcolors.to_rgb(col))
        dark = base * 0.6
        light = np.clip(base * 1.3, 0, 1)
        for k in range(40):
            full_colors.append(tuple(dark + (light - dark) * k / 39))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "segmented", full_colors, N=len(full_colors)
    )

    plt.figure(figsize=(8, 8))
    sns.heatmap(
        matrix,
        xticklabels=ranks,
        yticklabels=ranks,
        cmap=cmap,
        square=True,
        vmin=vmin,
        vmax=vmax,
        cbar=True
    )

    plt.title(f"Bucket {bucket_id} – {metric} Heatmap")
    plt.xlabel("Card 1")
    plt.ylabel("Card 2")

    for i in range(size):
        for j in range(size):
            if counts[i, j] > 0:
                plt.text(
                    j + 0.5, i + 0.5,
                    f"{matrix[i, j]:.2f}",
                    ha="center", va="center",
                    color="white", fontsize=7
                )

    patches = [
        mpatches.Patch(color=base_colors[i], label=labels[i])
        for i in range(len(base_colors))
    ]
    plt.legend(
        handles=patches,
        title=metric,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=5,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(PLOT_PATH /filename, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {filename}")

def gen_flop_plots(bucket_id = 0, filename = "flop_bucket_metrics.pkl"):
    with open(DATA_PATH / filename, "rb") as f:
        bucket_results = pickle.load(f)

    bucket_id = 0

    METRICS = {
        "EHS":     (0.0, 1.0),
        "VAR":     (0.0, 0.25),
        "MED":     (0.0, 1.0),
        "IQR":     (0.0, 1.0),
        "NEG_POT": (0.0, 1.0),
        "POS_POT": (0.0, 1.0),
        "CRASH":   (0.0, 1.0),
        "DOM":     (0.0, 1.0),
    }

    for metric, (vmin, vmax) in METRICS.items():
        plot_bucket_metric_heatmap(
            bucket_results=bucket_results,
            bucket_id=bucket_id,
            metric=metric,
            filename=f"flop_bucket_{bucket_id}_{metric.lower()}.png",
            vmin=vmin,
            vmax=vmax
        )
def gen_turn_plots(bucket_id = 0, filename = "turn_bucket_metrics.pkl"):
    with open(DATA_PATH / filename, "rb") as f:
        bucket_results = pickle.load(f)

    bucket_id = 0

    METRICS = {
        "EHS":     (0.0, 1.0),
        "VAR":     (0.0, 0.25),
        "MED":     (0.0, 1.0),
        "IQR":     (0.0, 1.0),
        "NEG_POT": (0.0, 1.0),
        "POS_POT": (0.0, 1.0),
        "CRASH":   (0.0, 1.0),
        "DOM":     (0.0, 1.0),
    }

    for metric, (vmin, vmax) in METRICS.items():
        plot_bucket_metric_heatmap(
            bucket_results=bucket_results,
            bucket_id=bucket_id,
            metric=metric,
            filename=f"turn_bucket_{bucket_id}_{metric.lower()}.png",
            vmin=vmin,
            vmax=vmax
        )


# In[4]:


if __name__ == "__main__":
    DATA_PATH = Path.cwd().parents[1] / "data"
    PLOT_PATH = Path.cwd().parents[1] / "plots"
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    RANKS = "23456789TJQKA"
    RANK_TO_INDEX = {r: i for i, r in enumerate(RANKS)}

    gen_preflop_plots(filename = "preflop_ehs.pkl")
    gen_flop_plots(bucket_id = 0, filename = "flop_bucket_metrics.pkl")
    gen_turn_plots(bucket_id = 0, filename = "turn_bucket_metrics.pkl")

