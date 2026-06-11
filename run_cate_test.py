"""
Test run: Stage 2 Causal Forest CATE estimation.
Loads the cleaned survey data, fits pairwise CausalForestDML models,
extracts CATEs, runs heterogeneity tests, and saves all outputs to
outputs/test_results/.
"""

import sys
import os

# Add project root to path so src.* and data.* imports resolve
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from src.preprocessing.data_cleaning import load_synthetic
from src.cate.causal_forest_epr import (
    run_cate_pipeline,
    plot_cate_distributions,
    plot_cate_by_subgroup,
    test_heterogeneity_hypotheses,
)

SAVE_DIR = os.path.join(ROOT, "outputs", "test_results")
os.makedirs(SAVE_DIR, exist_ok=True)

DATA_PATH = os.path.join(ROOT, "data", "processed", "survey_data_clean.csv")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading data...")
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded survey_data_clean.csv  ({len(df)} rows)")
else:
    print("  survey_data_clean.csv not found — generating synthetic data (n=600)")
    df, _ = load_synthetic(n=600, seed=42)

print(f"  Arm counts:\n{df['TREAT'].value_counts().sort_index().to_string()}")


# ---------------------------------------------------------------------------
# Stage 2: CATE estimation (pairwise Causal Forests)
# ---------------------------------------------------------------------------

models, df_cate, hypotheses = run_cate_pipeline(
    df,
    outcome="BAE_EPR",
    mode="pairwise",
    n_estimators=300,
    random_state=42,
)


# ---------------------------------------------------------------------------
# Save hypothesis test table
# ---------------------------------------------------------------------------

hyp_path = os.path.join(SAVE_DIR, "heterogeneity_hypothesis_tests.csv")
hypotheses.to_csv(hyp_path, index=False)
print(f"\nSaved: {hyp_path}")

# Also save CATE summary statistics per arm
cate_cols = [c for c in df_cate.columns if c.startswith("CATE_") and not c.endswith("_SE")]
cate_summary = df_cate[cate_cols].describe().round(4)
cate_summary_path = os.path.join(SAVE_DIR, "cate_summary_stats.csv")
cate_summary.to_csv(cate_summary_path)
print(f"Saved: {cate_summary_path}")

# CATE values with firm IDs for downstream SHAP
cate_values_path = os.path.join(SAVE_DIR, "cate_values.csv")
df_cate.to_csv(cate_values_path, index=False)
print(f"Saved: {cate_values_path}")


# ---------------------------------------------------------------------------
# Figure 1: CATE distributions (KDE)
# ---------------------------------------------------------------------------

fig1 = plot_cate_distributions(df_cate, figsize=(14, 4))
fig1.savefig(os.path.join(SAVE_DIR, "fig1_cate_distributions.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved: fig1_cate_distributions.png")


# ---------------------------------------------------------------------------
# Figure 2: CATE by key moderator subgroups (bar charts)
# ---------------------------------------------------------------------------

moderator_pairs = [
    ("CATE_REGULATORY",   "ENV_INSP",    "H2: ENV_INSP"),
    ("CATE_REPUTATIONAL", "EO_MEAN",     "H3: EO_MEAN"),
    ("CATE_MARKET",       "EXPORT_PCT",  "H1: EXPORT_PCT"),
]

fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
arm_colors = {"CATE_REGULATORY": "#e74c3c",
              "CATE_REPUTATIONAL": "#3498db",
              "CATE_MARKET": "#2ecc71"}

for ax, (cate_col, mod_col, title) in zip(axes, moderator_pairs):
    if cate_col not in df_cate.columns or mod_col not in df_cate.columns:
        ax.set_visible(False)
        continue

    df_plot = df_cate[[cate_col, mod_col]].dropna().copy()
    n_unique = df_plot[mod_col].nunique()

    if n_unique > 5:
        df_plot["group"] = pd.qcut(
            df_plot[mod_col], 5,
            labels=["Q1\n(Low)", "Q2", "Q3", "Q4", "Q5\n(High)"],
            duplicates="drop",
        )
    else:
        df_plot["group"] = df_plot[mod_col].map({0: "No (0)", 1: "Yes (1)"}).fillna(df_plot[mod_col].astype(str))

    agg = df_plot.groupby("group", observed=True)[cate_col].agg(
        mean="mean",
        sem=lambda x: x.std() / np.sqrt(len(x)),
    )

    ax.bar(range(len(agg)), agg["mean"],
           yerr=1.96 * agg["sem"], capsize=5,
           color=arm_colors.get(cate_col, "steelblue"), alpha=0.8)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg.index, fontsize=8)
    ax.axhline(0, linestyle="--", color="black", alpha=0.4)
    ax.set_title(f"{cate_col.replace('CATE_', '')}\n{title}", fontsize=9)
    ax.set_ylabel("Mean CATE (BAE_EPR, %)", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

fig2.suptitle("Mean CATE by Theory-Motivated Moderator Subgroups\n"
              "(bars = ±95% CI)", fontsize=11)
plt.tight_layout()
fig2.savefig(os.path.join(SAVE_DIR, "fig2_cate_by_moderator.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved: fig2_cate_by_moderator.png")


# ---------------------------------------------------------------------------
# Figure 3: CATE heterogeneity scatter — sorted individual CATEs
# ---------------------------------------------------------------------------

fig3, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
colors = ["#e74c3c", "#3498db", "#2ecc71"]
labels = ["Regulatory", "Reputational", "Market"]

for ax, col_suffix, color, label in zip(
    axes,
    ["REGULATORY", "REPUTATIONAL", "MARKET"],
    colors, labels
):
    col = f"CATE_{col_suffix}"
    se_col = f"CATE_{col_suffix}_SE"
    if col not in df_cate.columns:
        ax.set_visible(False)
        continue

    sorted_cate = df_cate[col].sort_values().reset_index(drop=True)
    ax.scatter(range(len(sorted_cate)), sorted_cate,
               color=color, alpha=0.4, s=6)
    ax.axhline(sorted_cate.mean(), linestyle="--", color="black",
               linewidth=1.5, label=f"Mean={sorted_cate.mean():.2f}%")
    ax.axhline(0, linestyle=":", color="gray", alpha=0.5)
    ax.set_title(f"{label} Pressure\nSorted Individual CATEs", fontsize=9)
    ax.set_xlabel("Firm rank", fontsize=8)
    ax.set_ylabel("Estimated CATE (BAE_EPR, %)", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

fig3.suptitle("Heterogeneity in Individual Treatment Effects by Governance Mechanism",
              fontsize=11)
plt.tight_layout()
fig3.savefig(os.path.join(SAVE_DIR, "fig3_sorted_individual_cates.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig3)
print("Saved: fig3_sorted_individual_cates.png")


# ---------------------------------------------------------------------------
# Figure 4: CATE quantile heatmap (top vs. bottom responders)
# ---------------------------------------------------------------------------

from data.synthetic.generate_epr_data import COVARIATE_COLS

key_vars = ["EXPORT_PCT", "ENV_INSP", "EO_MEAN", "CC_MEAN",
            "EMP_SIZE", "REVENUE", "RC_MEAN", "AWR_MEAN"]
avail_vars = [v for v in key_vars if v in df_cate.columns]

profiles = {}
for col_suffix, label in zip(["REGULATORY", "REPUTATIONAL", "MARKET"],
                              ["Regulatory", "Reputational", "Market"]):
    cate_col = f"CATE_{col_suffix}"
    if cate_col not in df_cate.columns:
        continue
    q75 = df_cate[cate_col].quantile(0.75)
    q25 = df_cate[cate_col].quantile(0.25)
    high = df_cate[df_cate[cate_col] >= q75][avail_vars].mean()
    low  = df_cate[df_cate[cate_col] <= q25][avail_vars].mean()
    profiles[label] = (high - low)

profile_df = pd.DataFrame(profiles, index=avail_vars)

fig4, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(profile_df.values, cmap="RdBu_r", aspect="auto",
               vmin=-profile_df.abs().max().max(),
               vmax=profile_df.abs().max().max())
ax.set_xticks(range(len(profile_df.columns)))
ax.set_xticklabels(profile_df.columns, fontsize=10)
ax.set_yticks(range(len(profile_df.index)))
ax.set_yticklabels(profile_df.index, fontsize=9)
for i in range(len(profile_df.index)):
    for j in range(len(profile_df.columns)):
        ax.text(j, i, f"{profile_df.values[i, j]:.2f}",
                ha="center", va="center", fontsize=8,
                color="black" if abs(profile_df.values[i, j]) < profile_df.abs().max().max() * 0.6 else "white")
plt.colorbar(im, ax=ax, label="High-responder mean − Low-responder mean")
ax.set_title("Covariate Profile: Top 25% vs. Bottom 25% CATE Responders\n"
             "(Red = high-responders score higher on this variable)", fontsize=10)
plt.tight_layout()
fig4.savefig(os.path.join(SAVE_DIR, "fig4_responder_profile_heatmap.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig4)
print("Saved: fig4_responder_profile_heatmap.png")


# ---------------------------------------------------------------------------
# Responder profile tables
# ---------------------------------------------------------------------------

for col_suffix, label in zip(["REGULATORY", "REPUTATIONAL", "MARKET"],
                              ["Regulatory", "Reputational", "Market"]):
    cate_col = f"CATE_{col_suffix}"
    if cate_col not in df_cate.columns:
        continue
    q75 = df_cate[cate_col].quantile(0.75)
    q25 = df_cate[cate_col].quantile(0.25)
    high = df_cate[df_cate[cate_col] >= q75][avail_vars].mean().rename("High Responder (Q75+)")
    low  = df_cate[df_cate[cate_col] <= q25][avail_vars].mean().rename("Low Responder (Q25-)")
    tbl = pd.concat([high, low], axis=1)
    tbl["Difference"] = tbl["High Responder (Q75+)"] - tbl["Low Responder (Q25-)"]
    tbl = tbl.round(3)
    path = os.path.join(SAVE_DIR, f"table_responder_profile_{label.lower()}.csv")
    tbl.to_csv(path)
    print(f"Saved: table_responder_profile_{label.lower()}.csv")


# ---------------------------------------------------------------------------
# Print final summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("TEST RUN COMPLETE")
print("=" * 72)
print(f"\nOutputs saved to: {SAVE_DIR}")
print("\nCate summary (mean CATE per arm):")
for col in cate_cols:
    label = col.replace("CATE_", "").title()
    print(f"  {label:14s}  mean={df_cate[col].mean():+6.3f}%  "
          f"sd={df_cate[col].std():5.3f}  "
          f"min={df_cate[col].min():+6.3f}  max={df_cate[col].max():+6.3f}")

print("\nHeterogeneity hypothesis tests:")
print(hypotheses.to_string(index=False))
