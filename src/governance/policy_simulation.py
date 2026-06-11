"""
Policy-targeting simulations: compare aggregate EPR compliance outcomes
under four assignment strategies.

Scenarios
---------
A  Enforcement for all firms
B  Reputation  for all firms
C  Market      for all firms
D  Each firm receives its optimal (highest-CATE) mechanism

The "expected gain" for a firm assigned treatment t is its CATE_t.
Aggregate gain = mean CATE across all firms under the assigned strategy.
Expected compliance = control-group baseline + aggregate gain.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

CATE_MAP = {
    "Enforcement": "CATE_REGULATORY",
    "Reputation":  "CATE_REPUTATIONAL",
    "Market":      "CATE_MARKET",
}

SCENARIO_LABELS = {
    "A_Enforcement":      "A: Universal Enforcement",
    "B_Reputation":       "B: Universal Reputation",
    "C_Market":           "C: Universal Market",
    "D_Optimal":          "D: Firm-Specific Optimal",
}


def simulate_scenarios(
    df_cate: pd.DataFrame,
    baseline_col: str = "BAE_EPR",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute aggregate expected compliance under Scenarios A-D.

    Parameters
    ----------
    df_cate      : DataFrame with CATE_* columns (output of CATE pipeline).
    baseline_col : Column for the baseline outcome (used to compute expected
                   absolute BAE_EPR, not just CATE gains).

    Returns
    -------
    scenario_df     : Summary with one row per scenario.
    assignment_df   : Scenario D assignment breakdown (n firms per mechanism).
    """
    avail = {k: v for k, v in CATE_MAP.items() if v in df_cate.columns}
    if not avail:
        raise ValueError("No CATE columns found. Run CATE pipeline first.")

    # Baseline: control-group mean of the outcome variable
    if "TREAT" in df_cate.columns and baseline_col in df_cate.columns:
        ctrl_mean = df_cate[df_cate["TREAT"] == 0][baseline_col].mean()
    elif baseline_col in df_cate.columns:
        ctrl_mean = df_cate[baseline_col].mean()
    else:
        ctrl_mean = np.nan

    labels   = list(avail.keys())
    cate_mat = df_cate[[avail[k] for k in labels]].values.astype(float)

    records = []

    # Scenarios A / B / C -- universal assignment
    for arm, col in avail.items():
        cates = df_cate[col].values
        records.append({
            "Scenario":           SCENARIO_LABELS.get(f"{'ABC'[labels.index(arm)]}_{arm}", arm),
            "Strategy":           f"Universal {arm}",
            "Mean CATE (pp)":     round(float(cates.mean()), 4),
            "SD of CATE":         round(float(cates.std()),  4),
            "Total CATE gain":    round(float(cates.sum()),  2),
            "Expected BAE_EPR":   round(ctrl_mean + cates.mean(), 4) if not np.isnan(ctrl_mean) else np.nan,
            "Firms assigned (%)": 100.0,
        })

    # Scenario D -- firm-specific optimal
    best_idx     = np.argmax(cate_mat, axis=1)
    optimal_cates = cate_mat[np.arange(len(df_cate)), best_idx]

    records.append({
        "Scenario":           SCENARIO_LABELS["D_Optimal"],
        "Strategy":           "Firm-Specific Optimal",
        "Mean CATE (pp)":     round(float(optimal_cates.mean()), 4),
        "SD of CATE":         round(float(optimal_cates.std()),  4),
        "Total CATE gain":    round(float(optimal_cates.sum()),  2),
        "Expected BAE_EPR":   round(ctrl_mean + optimal_cates.mean(), 4) if not np.isnan(ctrl_mean) else np.nan,
        "Firms assigned (%)": 100.0,
    })

    scenario_df = pd.DataFrame(records)

    # Gain of each scenario vs. the best universal strategy
    best_univ   = scenario_df.loc[
        scenario_df["Strategy"].str.startswith("Universal"), "Mean CATE (pp)"
    ].max()
    scenario_df["Gain vs. best universal (pp)"] = (
        scenario_df["Mean CATE (pp)"] - best_univ
    ).round(4)

    # Scenario D assignment breakdown
    assign_rows = []
    for i, label in enumerate(labels):
        mask = best_idx == i
        assign_rows.append({
            "Treatment":          label,
            "n_firms":            int(mask.sum()),
            "pct_firms":          round(float(mask.mean()) * 100, 1),
            "mean_optimal_cate":  round(float(cate_mat[mask, i].mean()), 4) if mask.sum() > 0 else np.nan,
        })
    assignment_df = pd.DataFrame(assign_rows)

    return scenario_df, assignment_df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_scenario_comparison(scenario_df: pd.DataFrame, figsize=(10, 5)):
    """Horizontal bar chart comparing mean CATE across all four scenarios."""
    import matplotlib.pyplot as plt

    color_map = {
        "Universal Enforcement":  "#e74c3c",
        "Universal Reputation":   "#3498db",
        "Universal Market":       "#2ecc71",
        "Firm-Specific Optimal":  "#9b59b6",
    }

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(scenario_df))
    bars = ax.barh(
        y,
        scenario_df["Mean CATE (pp)"].values,
        color=[color_map.get(s, "#888888") for s in scenario_df["Strategy"]],
        alpha=0.85, height=0.55,
    )

    # Annotate bars
    for bar, val, exp in zip(bars,
                              scenario_df["Mean CATE (pp)"].values,
                              scenario_df["Expected BAE_EPR"].values):
        label = f"  {val:.2f} pp"
        if not np.isnan(exp):
            label += f"  (expected BAE_EPR: {exp:.1f}%)"
        ax.text(bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(scenario_df["Scenario"], fontsize=9)
    ax.set_xlabel("Mean CATE -- expected gain in EPR budget share (pp)", fontsize=10)
    ax.set_title(
        "Policy Simulation: Aggregate Compliance Gain by Assignment Strategy\n"
        "Scenario D (optimal targeting) vs. uniform deployment",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3)

    # Reference line at best universal
    best_univ = scenario_df.loc[
        scenario_df["Strategy"].str.startswith("Universal"), "Mean CATE (pp)"
    ].max()
    ax.axvline(best_univ, linestyle="--", color="black", alpha=0.3, linewidth=1.2,
               label=f"Best universal ({best_univ:.2f} pp)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_optimal_assignment_map(
    df_cate: pd.DataFrame,
    x_col: str = "EXPORT_PCT",
    y_col: str = "EO_MEAN",
    figsize=(8, 6),
):
    """
    Scatter of firms in (x_col, y_col) space, coloured by optimal treatment.
    Reveals the policy-targeting surface along two key moderator dimensions.
    """
    import matplotlib.pyplot as plt

    avail  = {k: v for k, v in CATE_MAP.items() if v in df_cate.columns}
    labels = list(avail.keys())
    cate_mat = df_cate[[avail[k] for k in labels]].values.astype(float)
    best_idx = np.argmax(cate_mat, axis=1)

    arm_colors = {"Enforcement": "#e74c3c", "Reputation": "#3498db", "Market": "#2ecc71"}

    if x_col not in df_cate.columns or y_col not in df_cate.columns:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    for i, label in enumerate(labels):
        mask = best_idx == i
        ax.scatter(
            df_cate.loc[mask, x_col], df_cate.loc[mask, y_col],
            label=f"{label} (n={mask.sum()})",
            color=arm_colors.get(label, "#888888"),
            alpha=0.5, s=20,
        )

    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    ax.set_title(
        f"Optimal Governance Mechanism: {x_col} vs. {y_col}\n"
        f"(colour = which mechanism is best for each firm)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


def plot_assignment_breakdown(assignment_df: pd.DataFrame, figsize=(7, 4)):
    """Pie / donut chart of Scenario D treatment assignment shares."""
    import matplotlib.pyplot as plt

    colors = {"Enforcement": "#e74c3c", "Reputation": "#3498db", "Market": "#2ecc71"}
    fig, ax = plt.subplots(figsize=figsize)

    wedges, texts, autotexts = ax.pie(
        assignment_df["n_firms"],
        labels=assignment_df["Treatment"],
        colors=[colors.get(t, "#888888") for t in assignment_df["Treatment"]],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(10)

    # Donut hole
    centre = plt.Circle((0, 0), 0.45, color="white")
    ax.add_patch(centre)

    ax.set_title(
        "Scenario D: Share of Firms by Optimal Treatment\n"
        "(firm-specific governance targeting)",
        fontsize=11,
    )
    plt.tight_layout()
    return fig
