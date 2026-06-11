"""
Test run: Stage 3 SHAP analysis on top of Stage 2 Causal Forests.

SHAP strategy: EconML's CausalForestDML internally uses MultiOutputGRF
(Generalized Random Forest), which TreeSHAP does not support. We therefore
wrap model.effect(X) as a prediction function and use shap.PermutationExplainer,
which is model-agnostic and exact up to permutation-count precision.

Outputs saved to: outputs/test_results/shap/
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.preprocessing.data_cleaning import load_synthetic
from src.cate.causal_forest_epr import run_cate_pipeline
from data.synthetic.generate_epr_data import COVARIATE_COLS

SAVE_DIR = os.path.join(ROOT, "outputs", "test_results", "shap")
os.makedirs(SAVE_DIR, exist_ok=True)

DATA_PATH = os.path.join(ROOT, "data", "processed", "survey_data_clean.csv")

ARM_COLORS = {
    "Regulatory":   "#e74c3c",
    "Reputational": "#3498db",
    "Market":       "#2ecc71",
}

KEY_MODERATORS = {
    "Regulatory":   ["ENV_INSP", "CC_MEAN", "RP_MEAN", "AWR_MEAN", "ENV_PEN"],
    "Reputational": ["EO_MEAN",  "ESG_REP", "REP_MEAN", "EMP_SIZE", "EPR_REP"],
    "Market":       ["EXPORT_PCT", "SUPPLY_POS", "MP_MEAN", "REVENUE", "OWNERSHIP"],
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading data...")
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded survey_data_clean.csv  ({len(df)} rows)")
else:
    print("  Generating synthetic data (n=600)")
    df, _ = load_synthetic(n=600, seed=42)


# ---------------------------------------------------------------------------
# Stage 2: fit models
# ---------------------------------------------------------------------------

print("\nFitting causal forest models (Stage 2)...")
models, df_cate, _ = run_cate_pipeline(
    df, outcome="BAE_EPR", mode="pairwise",
    n_estimators=300, random_state=42,
)

X_cols = [c for c in COVARIATE_COLS if c in df.columns]
X = df_cate[X_cols].fillna(df_cate[X_cols].median()).values.astype(float)

# Smaller background for PermutationExplainer (100 representative rows)
rng = np.random.default_rng(42)
bg_idx = rng.choice(len(X), size=100, replace=False)
X_background = X[bg_idx]


# ---------------------------------------------------------------------------
# Stage 3: SHAP via PermutationExplainer (model-agnostic)
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("STAGE 3: SHAP ANALYSIS  (PermutationExplainer)")
print("=" * 72)

shap_dict = {}
for label, model in models.items():
    print(f"\n  [{label}] building explainer and computing SHAP values...")
    print(f"    (n={len(X)}, background={len(X_background)}, npermutations=50)")

    def make_predict(m):
        def predict(x):
            return m.effect(x).flatten()
        return predict

    explainer = shap.PermutationExplainer(
        make_predict(model),
        X_background,
        feature_names=X_cols,
    )
    sv = explainer(X, max_evals=500)
    shap_dict[label] = sv
    print(f"    SHAP values shape: {sv.values.shape}")


# ---------------------------------------------------------------------------
# Cross-arm importance table
# ---------------------------------------------------------------------------

print("\nCross-arm mean |SHAP| importance (top 15):")
imp_series = []
for label, sv in shap_dict.items():
    imp_series.append(pd.Series(
        np.abs(sv.values).mean(axis=0),
        index=X_cols, name=label,
    ))

importance_df = pd.concat(imp_series, axis=1)
importance_df["_max"] = importance_df.max(axis=1)
importance_df = (importance_df
                 .sort_values("_max", ascending=False)
                 .drop(columns="_max")
                 .head(15))

print(importance_df.round(5).to_string())
imp_path = os.path.join(SAVE_DIR, "table_cross_arm_shap_importance.csv")
importance_df.to_csv(imp_path)
print(f"\nSaved: {imp_path}")


# ---------------------------------------------------------------------------
# Figure 1: Cross-arm grouped bar chart
# ---------------------------------------------------------------------------

n_feat = len(importance_df)
n_arms = len(importance_df.columns)
bar_w  = 0.25
y_pos  = np.arange(n_feat)

fig1, ax = plt.subplots(figsize=(11, 7))
for i, arm in enumerate(importance_df.columns):
    offset = (i - (n_arms - 1) / 2) * bar_w
    ax.barh(
        y_pos + offset,
        importance_df[arm].values,
        height=bar_w,
        label=f"{arm} Pressure",
        color=ARM_COLORS.get(arm, "gray"),
        alpha=0.85,
    )
ax.set_yticks(y_pos)
ax.set_yticklabels(importance_df.index, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP value|  (impact on CATE)", fontsize=10)
ax.set_title(
    "Feature Importance by Governance Mechanism\n"
    "Which firm characteristics drive CATE heterogeneity?",
    fontsize=11,
)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig1.savefig(os.path.join(SAVE_DIR, "fig1_cross_arm_importance.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved: fig1_cross_arm_importance.png")


# ---------------------------------------------------------------------------
# Figure 2: Beeswarm plots — one per arm
# ---------------------------------------------------------------------------

for label, sv in shap_dict.items():
    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False, plot_size=None)
    plt.gcf().set_size_inches(9, 6)
    plt.suptitle(
        f"SHAP Feature Importance -- {label} Pressure\n"
        f"Impact on CATE: EPR budget allocation (BAE_EPR)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, f"fig2_beeswarm_{label.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved: fig2_beeswarm_{label.lower()}.png")


# ---------------------------------------------------------------------------
# Figure 3: Theory-motivated moderator dependence scatter panels
# ---------------------------------------------------------------------------

for label, sv in shap_dict.items():
    mods = [m for m in KEY_MODERATORS.get(label, []) if m in sv.feature_names]
    if not mods:
        continue

    n = len(mods)
    fig3, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, mods):
        feat_idx = sv.feature_names.index(feat)
        shap_vals = sv.values[:, feat_idx]
        feat_vals = sv.data[:, feat_idx]

        sc = ax.scatter(feat_vals, shap_vals,
                        c=feat_vals, cmap="RdBu_r", alpha=0.5, s=18,
                        vmin=np.percentile(feat_vals, 5),
                        vmax=np.percentile(feat_vals, 95))
        ax.axhline(0, linestyle="--", color="gray", alpha=0.5)

        # linear trend line
        if len(np.unique(feat_vals)) > 3:
            coef = np.polyfit(feat_vals, shap_vals, 1)
            xfit = np.linspace(feat_vals.min(), feat_vals.max(), 100)
            ax.plot(xfit, np.polyval(coef, xfit),
                    color="black", linewidth=1.5, alpha=0.7)

        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("Feature value", fontsize=8)
        ax.set_ylabel("SHAP (CATE impact)", fontsize=8)
        ax.grid(True, alpha=0.25)
        plt.colorbar(sc, ax=ax, pad=0.02)

    fig3.suptitle(
        f"Theory-Motivated Moderators -- {label} Pressure",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, f"fig3_moderators_{label.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved: fig3_moderators_{label.lower()}.png")


# ---------------------------------------------------------------------------
# Figure 4: Waterfall plots — median-CATE firm per arm
# ---------------------------------------------------------------------------

for label, sv in shap_dict.items():
    cate_col = f"CATE_{label.upper()}"
    if cate_col not in df_cate.columns:
        continue
    median_idx = (df_cate[cate_col] - df_cate[cate_col].median()).abs().idxmin()
    row_idx = df_cate.index.get_loc(median_idx)

    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(sv[row_idx], show=False, max_display=12)
    plt.title(
        f"SHAP Waterfall -- Median CATE Firm  [{label} Pressure]\n"
        f"CATE = {df_cate.loc[median_idx, cate_col]:.2f}%",
        fontsize=10,
    )
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, f"fig4_waterfall_{label.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved: fig4_waterfall_{label.lower()}.png")


# ---------------------------------------------------------------------------
# Figure 5: SHAP summary heatmap — all arms side by side
# ---------------------------------------------------------------------------

top_feats = importance_df.index.tolist()  # already sorted by max importance

fig5, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
for ax, (label, sv) in zip(axes, shap_dict.items()):
    feat_indices = [sv.feature_names.index(f) for f in top_feats if f in sv.feature_names]
    mat = sv.values[:, feat_indices]

    vmax = np.percentile(np.abs(mat), 98)
    im = ax.imshow(mat.T, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_title(f"{label} Pressure", fontsize=10)
    ax.set_yticks(range(len(feat_indices)))
    ax.set_yticklabels(top_feats[:len(feat_indices)], fontsize=8)
    ax.set_xlabel("Firm index", fontsize=8)
    plt.colorbar(im, ax=ax, label="SHAP value", pad=0.02)

fig5.suptitle(
    "SHAP Value Heatmap: Firm x Feature x Governance Mechanism\n"
    "(Red = pushes CATE up, Blue = pushes CATE down)",
    fontsize=11,
)
plt.tight_layout()
fig5.savefig(os.path.join(SAVE_DIR, "fig5_shap_heatmap.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig5)
print("Saved: fig5_shap_heatmap.png")


# ---------------------------------------------------------------------------
# Table: top-5 drivers per arm with direction
# ---------------------------------------------------------------------------

top5_records = []
for label, sv in shap_dict.items():
    mean_abs = pd.Series(np.abs(sv.values).mean(axis=0), index=X_cols)
    top5 = mean_abs.nlargest(5)
    for rank, (feat, val) in enumerate(top5.items(), 1):
        feat_idx = sv.feature_names.index(feat)
        corr = np.corrcoef(sv.data[:, feat_idx], sv.values[:, feat_idx])[0, 1]
        top5_records.append({
            "Arm":          label,
            "Rank":         rank,
            "Feature":      feat,
            "Mean |SHAP|":  round(val, 6),
            "Direction":    "+" if corr >= 0 else "-",
            "Interpretation": (
                f"Higher {feat} -> {'stronger' if corr >= 0 else 'weaker'} "
                f"{label.lower()} pressure effect"
            ),
        })

top5_df = pd.DataFrame(top5_records)
top5_path = os.path.join(SAVE_DIR, "table_top5_drivers_per_arm.csv")
top5_df.to_csv(top5_path, index=False)
print(f"Saved: {top5_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("SHAP ANALYSIS COMPLETE")
print("=" * 72)

print("\nTop 5 CATE drivers per governance mechanism:")
for arm in ["Regulatory", "Reputational", "Market"]:
    subset = top5_df[top5_df["Arm"] == arm]
    print(f"\n  {arm}:")
    for _, row in subset.iterrows():
        print(f"    {row['Rank']}. {row['Feature']:14s}  "
              f"mean|SHAP|={row['Mean |SHAP|']:.5f}  "
              f"({row['Direction']})  {row['Interpretation']}")

print(f"\nAll outputs: {SAVE_DIR}")
