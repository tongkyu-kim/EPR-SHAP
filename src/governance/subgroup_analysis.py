"""
Subgroup heterogeneity analysis: compare mean CATEs across firm
characteristic subgroups to answer "which firms respond to which mechanism?"

Subgroups
---------
- Export intensity : EXPORT_PCT median split -> low / high exporters
- ESG reporting    : ESG_REP binary
- Prior inspection : ENV_INSP binary
- Firm size        : EMP_SIZE (1-6 ordinal categories)
- Supply chain pos : SUPPLY_POS (1-6 ordinal)
- ISO 14001        : ISO14001 binary
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional

CATE_MAP = {
    "Enforcement":  "CATE_REGULATORY",
    "Reputation":   "CATE_REPUTATIONAL",
    "Market":       "CATE_MARKET",
}

SUBGROUP_SPECS = {
    "Export intensity": {
        "col": "EXPORT_PCT", "type": "binary_median",
        "labels": {0: "Low export", 1: "High export"},
    },
    "ESG reporting": {
        "col": "ESG_REP", "type": "binary",
        "labels": {0: "No ESG", 1: "ESG reporter"},
    },
    "Prior inspection": {
        "col": "ENV_INSP", "type": "binary",
        "labels": {0: "Not inspected", 1: "Inspected"},
    },
    "ISO 14001": {
        "col": "ISO14001", "type": "binary",
        "labels": {0: "Not certified", 1: "ISO 14001"},
    },
    "Firm size": {
        "col": "EMP_SIZE", "type": "ordinal",
        "map": {1: "Micro", 2: "Small", 3: "Medium",
                4: "Large", 5: "Very large", 6: "Giant"},
    },
    "Supply chain position": {
        "col": "SUPPLY_POS", "type": "ordinal",
        "map": {1: "Tier-3", 2: "Tier-2", 3: "Tier-1",
                4: "Assembler", 5: "Brand", 6: "Retailer"},
    },
}


def _group_labels(df: pd.DataFrame, spec: dict) -> Optional[pd.Series]:
    col = spec["col"]
    if col not in df.columns:
        return None
    if spec["type"] == "binary":
        return df[col].map(spec["labels"])
    if spec["type"] == "binary_median":
        med = df[col].median()
        return (df[col] >= med).astype(int).map(spec["labels"])
    if spec["type"] == "ordinal":
        return df[col].map(spec["map"]).fillna("Other")
    return None


# ---------------------------------------------------------------------------
# Main tables
# ---------------------------------------------------------------------------

def subgroup_cate_table(df_cate: pd.DataFrame) -> pd.DataFrame:
    """
    Long-format table: mean CATE ± SE for every subgroup × arm combination.
    """
    avail = {k: v for k, v in CATE_MAP.items() if v in df_cate.columns}
    rows = []

    for sg_name, spec in SUBGROUP_SPECS.items():
        groups = _group_labels(df_cate, spec)
        if groups is None:
            continue
        df_sg = df_cate.copy()
        df_sg["_g"] = groups

        for arm, col in avail.items():
            agg = df_sg.groupby("_g")[col].agg(
                mean="mean",
                se=lambda x: x.std() / np.sqrt(len(x)),
                n="count",
            )
            for g_label, r in agg.iterrows():
                rows.append({
                    "Subgroup":   sg_name,
                    "Group":      g_label,
                    "Arm":        arm,
                    "Mean CATE":  round(r["mean"], 4),
                    "SE":         round(r["se"],   4),
                    "N":          int(r["n"]),
                })

    return pd.DataFrame(rows)


def subgroup_ttest_table(df_cate: pd.DataFrame) -> pd.DataFrame:
    """
    Welch t-test for binary / median-split subgroups (high vs. low group).
    Returns t-stat, p-value, and difference in mean CATE per arm.
    """
    avail = {k: v for k, v in CATE_MAP.items() if v in df_cate.columns}
    binary_specs = {k: v for k, v in SUBGROUP_SPECS.items()
                    if v["type"] in ("binary", "binary_median")}
    rows = []

    for sg_name, spec in binary_specs.items():
        col_sg = spec["col"]
        if col_sg not in df_cate.columns:
            continue
        if spec["type"] == "binary_median":
            mask_high = df_cate[col_sg] >= df_cate[col_sg].median()
        else:
            mask_high = df_cate[col_sg] == 1

        for arm, cate_col in avail.items():
            high = df_cate[mask_high][cate_col].dropna()
            low  = df_cate[~mask_high][cate_col].dropna()
            t, p = stats.ttest_ind(high, low, equal_var=False)
            rows.append({
                "Subgroup":          sg_name,
                "Arm":               arm,
                "Mean CATE (high)":  round(high.mean(), 4),
                "Mean CATE (low)":   round(low.mean(),  4),
                "Difference":        round(high.mean() - low.mean(), 4),
                "t_stat":            round(float(t), 4),
                "p_value":           round(float(p), 4),
                "Sig":               ("***" if p < 0.001 else "**" if p < 0.01
                                      else "*" if p < 0.05 else ""),
                "n_high": len(high),
                "n_low":  len(low),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_subgroup_heatmap(subgroup_df: pd.DataFrame, figsize=(13, 9)):
    """
    Heatmap: rows = subgroup-category, columns = treatment arms.
    Cell = mean CATE; colour ramp shows which groups respond most strongly.
    """
    import matplotlib.pyplot as plt

    pivot = subgroup_df.pivot_table(
        index=["Subgroup", "Group"],
        columns="Arm",
        values="Mean CATE",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=figsize)
    vmax = pivot.abs().max().max() * 1.2
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=vmax * 0.1, vmax=vmax)

    ylabels = [f"{sg}  |  {grp}" for sg, grp in pivot.index]
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10, fontweight="bold")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Mean CATE (BAE_EPR, pp)")
    ax.set_title(
        "Mean Treatment Effect by Firm Characteristic and Governance Mechanism\n"
        "(Green = stronger treatment response)",
        fontsize=11,
    )
    plt.tight_layout()
    return fig


def plot_subgroup_bar_panel(subgroup_df: pd.DataFrame, figsize=(16, 11)):
    """
    Panel of bar charts — one subplot per subgroup variable.
    Grouped bars show mean CATE per arm within each subgroup category.
    """
    import matplotlib.pyplot as plt

    arm_colors = {"Enforcement": "#e74c3c", "Reputation": "#3498db", "Market": "#2ecc71"}
    subgroups = subgroup_df["Subgroup"].unique()
    n = len(subgroups)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if n > 1 else [axes]

    for ax, sg in zip(axes_flat, subgroups):
        sub  = subgroup_df[subgroup_df["Subgroup"] == sg]
        groups = sub["Group"].unique()
        arms   = [a for a in arm_colors if a in sub["Arm"].unique()]
        bar_w  = 0.25
        x = np.arange(len(groups))

        for i, arm in enumerate(arms):
            arm_data = sub[sub["Arm"] == arm].set_index("Group")
            means = [arm_data.loc[g, "Mean CATE"] if g in arm_data.index else 0.0
                     for g in groups]
            ses   = [arm_data.loc[g, "SE"]        if g in arm_data.index else 0.0
                     for g in groups]
            offset = (i - (len(arms) - 1) / 2) * bar_w
            ax.bar(x + offset, means, bar_w,
                   label=arm, yerr=[1.96 * s for s in ses], capsize=3,
                   color=arm_colors[arm], alpha=0.85, error_kw={"linewidth": 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=7, rotation=25, ha="right")
        ax.set_title(sg, fontsize=9, fontweight="bold")
        ax.set_ylabel("Mean CATE (%)", fontsize=8)
        ax.axhline(0, linestyle="--", color="gray", alpha=0.4)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes_flat[0]:
            ax.legend(fontsize=7)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "Treatment Effects by Firm Characteristics (bars = mean CATE +/- 95% CI)",
        fontsize=12,
    )
    plt.tight_layout()
    return fig
