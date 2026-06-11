"""
Shared plotting utilities for the EPR governance experiment.

Provides consistent styling and reusable figure helpers used across
the ATE, CATE, and SHAP analysis modules.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional, List


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

ARM_COLORS = {
    "Regulatory":   "#e74c3c",
    "Reputational": "#3498db",
    "Market":       "#2ecc71",
    "Control":      "#95a5a6",
}

plt.rcParams.update({
    "figure.dpi":       130,
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
})


# ---------------------------------------------------------------------------
# ATE / coefficient plots
# ---------------------------------------------------------------------------

def plot_coefplot(
    results_df: pd.DataFrame,
    outcome_label: str = "",
    figsize=(7, 3.5),
) -> plt.Figure:
    """
    Coefficient plot (dot + 95% CI) for ATE results from ate_estimation.

    Expects columns: arm, coef, ci_lower, ci_upper
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, row in results_df.iterrows():
        color = ARM_COLORS.get(row["arm"], "gray")
        ax.plot([row["ci_lower"], row["ci_upper"]], [i, i],
                color=color, linewidth=2, alpha=0.8)
        ax.scatter(row["coef"], i, color=color, s=80, zorder=5)

    ax.axvline(0, linestyle="--", color="black", linewidth=0.8, alpha=0.6)
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df["arm"])
    ax.set_xlabel("ATE Estimate (95% CI)")
    ax.set_title(f"Average Treatment Effects\n{outcome_label}")
    plt.tight_layout()
    return fig


def plot_multi_outcome_coefplot(
    results: dict,
    figsize=(12, 4),
) -> plt.Figure:
    """
    Panel of coefficient plots, one per outcome.
    results: dict from estimate_all_ates()
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (outcome, df_res) in zip(axes, results.items()):
        label = df_res["outcome"].iloc[0] if "outcome" in df_res.columns else outcome
        for i, row in df_res.iterrows():
            color = ARM_COLORS.get(row["arm"], "gray")
            ax.plot([row["ci_lower"], row["ci_upper"]], [i, i],
                    color=color, linewidth=2, alpha=0.8)
            ax.scatter(row["coef"], i, color=color, s=60, zorder=5,
                       label=row["arm"] if i == df_res.index[0] else "")
        ax.axvline(0, linestyle="--", color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(label, fontsize=9)
        ax.set_yticks(range(len(df_res)))
        ax.set_yticklabels(df_res["arm"])

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Average Treatment Effects by Outcome", fontsize=12)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Balance check visualisation
# ---------------------------------------------------------------------------

def plot_balance(balance_df: pd.DataFrame, figsize=(8, 0.35)) -> plt.Figure:
    """
    Lollipop plot of p-values from the balance check.
    Covariates below p=0.05 are highlighted in red.
    """
    n = len(balance_df)
    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.35)))

    for i, row in balance_df.iterrows():
        color = "#e74c3c" if not row["balanced"] else "#2ecc71"
        ax.hlines(i, 0, row["p_value"], color=color, linewidth=1.5, alpha=0.7)
        ax.scatter(row["p_value"], i, color=color, s=50)

    ax.axvline(0.05, linestyle="--", color="black", linewidth=0.8,
               label="p = 0.05 threshold")
    ax.set_yticks(range(n))
    ax.set_yticklabels(balance_df["variable"], fontsize=8)
    ax.set_xlabel("p-value")
    ax.set_title("Randomisation Balance Check\n(red = p < 0.05; indicates imbalance)")
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CATE distribution comparison
# ---------------------------------------------------------------------------

def plot_cate_violin(df_cate: pd.DataFrame, figsize=(9, 5)) -> plt.Figure:
    """
    Violin plot comparing CATE distributions across the three arms.
    """
    cate_cols = {
        "cate_regulatory":    "Regulatory",
        "cate_reputational":  "Reputational",
        "cate_market":        "Market",
    }
    available = {v: df_cate[k].dropna().values
                 for k, v in cate_cols.items() if k in df_cate.columns}

    if not available:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    positions = list(range(1, len(available) + 1))
    labels = list(available.keys())
    data   = list(available.values())

    parts = ax.violinplot(data, positions=positions, showmedians=True, showmeans=False)
    for i, (body, label) in enumerate(zip(parts["bodies"], labels)):
        body.set_facecolor(ARM_COLORS.get(label, "gray"))
        body.set_alpha(0.7)

    ax.axhline(0, linestyle="--", color="black", linewidth=0.8, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Estimated CATE")
    ax.set_title("Distribution of Heterogeneous Treatment Effects\nby Governance Mechanism")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, path: str, dpi: int = 180) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def latex_table(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Convert a DataFrame to a LaTeX booktabs table string."""
    return df.to_latex(
        index=False,
        caption=caption,
        label=label,
        escape=True,
        float_format="%.4f",
        column_format="l" + "r" * (len(df.columns) - 1),
    )
