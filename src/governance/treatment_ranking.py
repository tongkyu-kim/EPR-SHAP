"""
Treatment ranking: for every firm, identify which governance mechanism
produces the highest CATE and quantify the gain from optimal targeting.

Outputs
-------
df_ranked      : original df + best_treatment, best_cate, gain_over_mean,
                 gain_over_worst, rank_enforcement/reputation/market
ranking_summary: frequency table of optimal-treatment assignments
"""

import numpy as np
import pandas as pd
from typing import Optional

# Maps user-facing arm names -> CATE column names (from causal_forest_epr.py)
CATE_MAP = {
    "Enforcement":  "CATE_REGULATORY",
    "Reputation":   "CATE_REPUTATIONAL",
    "Market":       "CATE_MARKET",
}


def compute_treatment_ranking(df_cate: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate each firm with its optimal governance mechanism.

    Added columns
    -------------
    best_treatment   : Enforcement | Reputation | Market
    best_cate        : CATE value of the optimal mechanism
    gain_over_mean   : best_cate - mean(all three CATEs)
    gain_over_worst  : max_cate - min_cate (spread across mechanisms)
    rank_enforcement : rank of enforcement (1 = best, 3 = worst)
    rank_reputation  : rank of reputation
    rank_market      : rank of market
    """
    avail = {k: v for k, v in CATE_MAP.items() if v in df_cate.columns}
    if not avail:
        raise ValueError("No CATE columns found in df_cate. Run CATE pipeline first.")

    labels = list(avail.keys())
    cate_mat = df_cate[[avail[k] for k in labels]].values.astype(float)

    best_idx = np.argmax(cate_mat, axis=1)

    df = df_cate.copy()
    df["best_treatment"]  = [labels[i] for i in best_idx]
    df["best_cate"]       = cate_mat[np.arange(len(df)), best_idx]
    df["gain_over_mean"]  = df["best_cate"] - cate_mat.mean(axis=1)
    df["gain_over_worst"] = cate_mat.max(axis=1) - cate_mat.min(axis=1)

    # Ranks: position in descending-sorted order (1 = highest CATE)
    # argsort returns indices that would sort ascending; flip for descending
    sort_desc = np.argsort(-cate_mat, axis=1)          # shape (n, 3)
    rank_mat  = np.empty_like(sort_desc)
    for row in range(len(df)):
        for rank, col_idx in enumerate(sort_desc[row]):
            rank_mat[row, col_idx] = rank + 1

    for j, label in enumerate(labels):
        df[f"rank_{label.lower()}"] = rank_mat[:, j]

    return df


def ranking_summary(df_ranked: pd.DataFrame) -> pd.DataFrame:
    """
    Frequency table of best-treatment assignments.

    Returns DataFrame with columns:
        n_firms, pct_firms, mean_best_cate, mean_gain_over_mean
    """
    counts = df_ranked["best_treatment"].value_counts()
    pct    = (counts / len(df_ranked) * 100).round(1)

    summary = pd.DataFrame({
        "n_firms":            counts,
        "pct_firms":          pct,
        "mean_best_cate":     df_ranked.groupby("best_treatment")["best_cate"].mean().round(4),
        "mean_gain_over_mean": df_ranked.groupby("best_treatment")["gain_over_mean"].mean().round(4),
    }).sort_values("n_firms", ascending=False)

    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_ranking_frequencies(df_ranked: pd.DataFrame, figsize=(8, 5)):
    """Bar chart: how many firms are optimally targeted by each mechanism."""
    import matplotlib.pyplot as plt

    counts = df_ranked["best_treatment"].value_counts()
    colors = {"Enforcement": "#e74c3c", "Reputation": "#3498db", "Market": "#2ecc71"}

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[colors.get(k, "#888888") for k in counts.index],
        alpha=0.85, edgecolor="white",
    )
    for bar, val in zip(bars, counts.values):
        pct = val / len(df_ranked) * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{pct:.1f}%", ha="center", fontsize=10)

    ax.set_ylabel("Number of firms", fontsize=10)
    ax.set_title(
        "Optimal Governance Mechanism by Firm\n"
        "(Which mechanism maximises expected CATE?)",
        fontsize=11,
    )
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_cate_pairwise_scatter(df_ranked: pd.DataFrame, figsize=(13, 5)):
    """
    Two scatter plots: (Enforcement vs Reputation) and (Enforcement vs Market).
    Points coloured by best-treatment assignment.
    """
    import matplotlib.pyplot as plt

    colors = {"Enforcement": "#e74c3c", "Reputation": "#3498db", "Market": "#2ecc71"}
    pairs = [
        ("CATE_REGULATORY",   "CATE_REPUTATIONAL",  "Enforcement CATE (%)", "Reputation CATE (%)"),
        ("CATE_REGULATORY",   "CATE_MARKET",         "Enforcement CATE (%)", "Market CATE (%)"),
    ]
    avail_pairs = [(a, b, la, lb) for a, b, la, lb in pairs
                   if a in df_ranked.columns and b in df_ranked.columns]
    if not avail_pairs:
        return None

    fig, axes = plt.subplots(1, len(avail_pairs), figsize=figsize)
    if len(avail_pairs) == 1:
        axes = [axes]

    for ax, (cx, cy, lx, ly) in zip(axes, avail_pairs):
        for treat, grp in df_ranked.groupby("best_treatment"):
            ax.scatter(grp[cx], grp[cy],
                       label=treat, alpha=0.45, s=18,
                       color=colors.get(treat, "#888888"))
        lo = min(df_ranked[cx].min(), df_ranked[cy].min())
        hi = max(df_ranked[cx].max(), df_ranked[cy].max())
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.35, linewidth=1, label="x=y")
        ax.set_xlabel(lx, fontsize=9)
        ax.set_ylabel(ly, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle("CATE Comparison by Optimal Treatment Assignment", fontsize=11)
    plt.tight_layout()
    return fig


def plot_gain_distribution(df_ranked: pd.DataFrame, figsize=(9, 4)):
    """KDE of gain_over_mean — how much does optimal targeting help each firm?"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    df_ranked["gain_over_mean"].plot.kde(ax=axes[0], color="#9b59b6", linewidth=2)
    axes[0].axvline(0, linestyle="--", color="gray", alpha=0.5)
    axes[0].set_title("Gain from Optimal vs. Mean Treatment", fontsize=10)
    axes[0].set_xlabel("CATE gain (pp)", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    df_ranked["gain_over_worst"].plot.kde(ax=axes[1], color="#e67e22", linewidth=2)
    axes[1].axvline(0, linestyle="--", color="gray", alpha=0.5)
    axes[1].set_title("Spread: Best vs. Worst Treatment per Firm", fontsize=10)
    axes[1].set_xlabel("CATE spread (pp)", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Value of Firm-Specific Governance Targeting", fontsize=11)
    plt.tight_layout()
    return fig
