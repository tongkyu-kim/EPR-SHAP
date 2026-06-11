"""
Stage 3: SHAP-based explanation of CATE heterogeneity.

Goal: "Which firm characteristics drive responsiveness to each
      governance mechanism?"

Strategy
--------
EconML's CausalForestDML uses MultiOutputGRF internally, which is not
supported by TreeSHAP. We therefore wrap each model's .effect() method
as a prediction function and use shap.PermutationExplainer, which is
model-agnostic (exact up to permutation count).

Per-arm analysis
----------------
A separate PermutationExplainer is built for each pairwise model.
SHAP values explain variation in CATE, not in the raw outcome.

Theory-motivated moderators
----------------------------
Enforcement  : ENV_INSP, CC_MEAN, RP_MEAN, AWR_MEAN, ENV_PEN
Reputation   : EO_MEAN,  ESG_REP, REP_MEAN, EMP_SIZE, EPR_REP
Market       : EXPORT_PCT, SUPPLY_POS, MP_MEAN, REVENUE, OWNERSHIP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Dict, List, Optional

from econml.dml import CausalForestDML
from data.synthetic.generate_epr_data import COVARIATE_COLS

ARMS = {1: "Regulatory", 2: "Reputational", 3: "Market"}

# Maps governance arm names to CATE column names
ARM_CATE_COL = {
    "Regulatory":   "CATE_REGULATORY",
    "Reputational": "CATE_REPUTATIONAL",
    "Market":       "CATE_MARKET",
}

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

DEFAULT_COVARIATES = COVARIATE_COLS


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: CausalForestDML,
    X: np.ndarray,
    X_background: np.ndarray,
    feature_names: List[str],
    max_evals: int = 500,
) -> shap.Explanation:
    """
    Compute PermutationSHAP values for one governance arm.

    The prediction function is model.effect(X).flatten(), so SHAP values
    explain variation in the CATE (not the raw outcome).

    Parameters
    ----------
    X_background : smaller reference dataset (~100 rows) used by the
                   explainer to marginalise over features.
    max_evals    : number of model calls per explained instance.
                   Higher = more accurate but slower.
                   Default 500 ~ 13 permutations for 17 features.
    """

    def predict_cate(x):
        return model.effect(x).flatten()

    explainer = shap.PermutationExplainer(
        predict_cate,
        X_background,
        feature_names=feature_names,
    )
    sv = explainer(X, max_evals=max_evals)
    return sv


def compute_all_shap_values(
    models: Dict[str, CausalForestDML],
    df: pd.DataFrame,
    covariates: Optional[List[str]] = None,
    bg_size: int = 100,
    max_evals: int = 500,
    random_state: int = 42,
) -> Dict[str, shap.Explanation]:
    """
    Compute SHAP values for every pairwise arm model.

    Returns  {arm_label: shap.Explanation}
    """
    covariates = covariates or DEFAULT_COVARIATES
    X_cols = [c for c in covariates if c in df.columns]
    X      = df[X_cols].fillna(df[X_cols].median()).values.astype(float)

    rng = np.random.default_rng(random_state)
    bg_idx = rng.choice(len(X), size=min(bg_size, len(X)), replace=False)
    X_bg   = X[bg_idx]

    shap_dict = {}
    for label, model in models.items():
        print(f"  Computing SHAP [{label}]  "
              f"(n={len(X)}, bg={len(X_bg)}, max_evals={max_evals})...")
        sv = compute_shap_values(model, X, X_bg, X_cols, max_evals)
        shap_dict[label] = sv
        print(f"    Done. Shape: {sv.values.shape}")

    return shap_dict


# ---------------------------------------------------------------------------
# Importance summary
# ---------------------------------------------------------------------------

def cross_arm_importance(
    shap_dict: Dict[str, shap.Explanation],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Mean |SHAP| per feature × arm, sorted by max importance across arms.
    """
    series_list = []
    for label, sv in shap_dict.items():
        s = pd.Series(np.abs(sv.values).mean(axis=0),
                      index=sv.feature_names, name=label)
        series_list.append(s)

    imp = pd.concat(series_list, axis=1)
    imp["_max"] = imp.max(axis=1)
    return (imp.sort_values("_max", ascending=False)
               .drop(columns="_max")
               .head(top_n))


def top_drivers_table(
    shap_dict: Dict[str, shap.Explanation],
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Top N SHAP drivers per arm with directional interpretation.
    """
    rows = []
    for label, sv in shap_dict.items():
        mean_abs = pd.Series(np.abs(sv.values).mean(axis=0),
                             index=sv.feature_names)
        for rank, (feat, val) in enumerate(mean_abs.nlargest(top_n).items(), 1):
            feat_idx = list(sv.feature_names).index(feat)
            corr = np.corrcoef(sv.data[:, feat_idx],
                               sv.values[:, feat_idx])[0, 1]
            rows.append({
                "Arm":          label,
                "Rank":         rank,
                "Feature":      feat,
                "Mean |SHAP|":  round(val, 6),
                "Direction":    "+" if corr >= 0 else "-",
                "Interpretation": (
                    f"Higher {feat} -> "
                    f"{'stronger' if corr >= 0 else 'weaker'} "
                    f"{label.lower()} effect"
                ),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisation: importance
# ---------------------------------------------------------------------------

def plot_cross_arm_importance(
    importance_df: pd.DataFrame,
    figsize=(11, 7),
) -> plt.Figure:
    """Grouped horizontal bar chart: feature importance by governance arm."""
    n_feat = len(importance_df)
    n_arms = len(importance_df.columns)
    bar_w  = 0.25
    y_pos  = np.arange(n_feat)

    fig, ax = plt.subplots(figsize=figsize)
    for i, arm in enumerate(importance_df.columns):
        offset = (i - (n_arms - 1) / 2) * bar_w
        ax.barh(
            y_pos + offset,
            importance_df[arm].values,
            height=bar_w,
            label=f"{arm} Pressure",
            color=ARM_COLORS.get(arm, "#888888"),
            alpha=0.85,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|  (impact on CATE)", fontsize=10)
    ax.set_title(
        "SHAP Feature Importance by Governance Mechanism\n"
        "Which firm characteristics drive CATE heterogeneity?",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_beeswarm(
    sv: shap.Explanation,
    arm_label: str,
    max_display: int = 15,
) -> plt.Figure:
    """
    SHAP beeswarm plot for one governance arm.
    Y-axis: features ranked by mean |SHAP|.
    Colour: feature value (red = high, blue = low).
    """
    plt.figure()
    shap.plots.beeswarm(sv, max_display=max_display, show=False, plot_size=None)
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    fig.suptitle(
        f"SHAP Feature Importance -- {arm_label} Pressure\n"
        "Impact on CATE (EPR budget allocation, BAE_EPR)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Visualisation: dependence / moderators
# ---------------------------------------------------------------------------

def plot_moderator_dependence(
    sv: shap.Explanation,
    arm_label: str,
    figsize_per_panel=(3.5, 4),
) -> Optional[plt.Figure]:
    """
    Panel of SHAP dependence scatter plots for theory-motivated moderators.
    One panel per moderator; linear trend overlaid.
    """
    mods = [m for m in KEY_MODERATORS.get(arm_label, [])
            if m in sv.feature_names]
    if not mods:
        return None

    n   = len(mods)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n,
                                             figsize_per_panel[1]))
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, mods):
        feat_idx  = list(sv.feature_names).index(feat)
        shap_vals = sv.values[:, feat_idx]
        feat_vals = sv.data[:,  feat_idx]

        sc = ax.scatter(feat_vals, shap_vals,
                        c=feat_vals, cmap="RdBu_r", alpha=0.5, s=16,
                        vmin=np.percentile(feat_vals, 5),
                        vmax=np.percentile(feat_vals, 95))

        # Linear trend
        if len(np.unique(feat_vals)) > 3:
            coef = np.polyfit(feat_vals, shap_vals, 1)
            xfit = np.linspace(feat_vals.min(), feat_vals.max(), 100)
            ax.plot(xfit, np.polyval(coef, xfit),
                    color="black", linewidth=1.5, alpha=0.7)

        ax.axhline(0, linestyle="--", color="gray", alpha=0.5)
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("Feature value", fontsize=8)
        ax.set_ylabel("SHAP (CATE impact)", fontsize=8)
        ax.grid(True, alpha=0.25)
        plt.colorbar(sc, ax=ax, pad=0.02)

    fig.suptitle(
        f"Theory-Motivated Moderators -- {arm_label} Pressure",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_waterfall(
    sv: shap.Explanation,
    idx: int,
    arm_label: str,
    cate_value: Optional[float] = None,
) -> plt.Figure:
    """Waterfall plot for a single firm (default: median-CATE firm)."""
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(sv[idx], show=False, max_display=12)
    title = f"SHAP Waterfall -- {arm_label} Pressure  [Firm #{idx}]"
    if cate_value is not None:
        title += f"\nCATe = {cate_value:.2f}%"
    plt.title(title, fontsize=10)
    plt.tight_layout()
    return plt.gcf()


def plot_shap_heatmap(
    shap_dict: Dict[str, shap.Explanation],
    importance_df: pd.DataFrame,
    figsize=(15, 7),
) -> plt.Figure:
    """
    Firm x Feature SHAP heatmap, one subplot per arm.
    Red = pushes CATE up; Blue = pushes CATE down.
    """
    top_feats = importance_df.index.tolist()
    fig, axes = plt.subplots(1, len(shap_dict), figsize=figsize, sharey=True)
    if len(shap_dict) == 1:
        axes = [axes]

    for ax, (label, sv) in zip(axes, shap_dict.items()):
        feat_indices = [list(sv.feature_names).index(f)
                        for f in top_feats if f in sv.feature_names]
        mat  = sv.values[:, feat_indices]
        vmax = np.percentile(np.abs(mat), 98)

        im = ax.imshow(mat.T, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(f"{label} Pressure", fontsize=10)
        ax.set_yticks(range(len(feat_indices)))
        ax.set_yticklabels(top_feats[:len(feat_indices)], fontsize=8)
        ax.set_xlabel("Firm index (sorted by row)", fontsize=8)
        plt.colorbar(im, ax=ax, label="SHAP value", pad=0.02)

    fig.suptitle(
        "SHAP Value Heatmap: Firm x Feature x Governance Mechanism\n"
        "(Red = increases CATE, Blue = decreases CATE)",
        fontsize=11,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Full SHAP pipeline
# ---------------------------------------------------------------------------

def run_shap_pipeline(
    models: Dict[str, CausalForestDML],
    df_cate: pd.DataFrame,
    covariates: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    bg_size: int = 100,
    max_evals: int = 500,
) -> Dict:
    """
    End-to-end SHAP analysis: compute -> visualise -> export.

    Parameters
    ----------
    models    : Pairwise fitted models from run_cate_pipeline().
    df_cate   : DataFrame with CATE_* columns.
    save_dir  : If given, all figures and tables are saved here.

    Returns
    -------
    dict with keys: shap_values, importance_df, top_drivers, figures
    """
    print("=" * 72)
    print("STAGE 3: SHAP ANALYSIS")
    print("=" * 72)

    covariates = covariates or DEFAULT_COVARIATES

    print("\nComputing SHAP values (PermutationExplainer)...")
    shap_dict = compute_all_shap_values(
        models, df_cate, covariates,
        bg_size=bg_size, max_evals=max_evals,
    )

    print("\nCross-arm importance (top 15):")
    importance_df = cross_arm_importance(shap_dict, top_n=15)
    print(importance_df.round(5).to_string())

    drivers = top_drivers_table(shap_dict)

    figures = {}

    figures["cross_arm_importance"] = plot_cross_arm_importance(importance_df)
    figures["shap_heatmap"]         = plot_shap_heatmap(shap_dict, importance_df)

    for label, sv in shap_dict.items():
        figures[f"beeswarm_{label}"]   = plot_beeswarm(sv, label)
        figures[f"moderators_{label}"] = plot_moderator_dependence(sv, label)

        cate_col = ARM_CATE_COL.get(label)
        if cate_col and cate_col in df_cate.columns:
            med_idx = (df_cate[cate_col] - df_cate[cate_col].median()).abs().idxmin()
            row_idx = df_cate.index.get_loc(med_idx)
            figures[f"waterfall_{label}"] = plot_waterfall(
                sv, row_idx, label, float(df_cate.loc[med_idx, cate_col])
            )

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)

        importance_df.to_csv(os.path.join(save_dir, "shap_cross_arm_importance.csv"))
        drivers.to_csv(os.path.join(save_dir, "shap_top_drivers.csv"), index=False)

        for name, fig in figures.items():
            if fig is not None:
                path = os.path.join(save_dir, f"{name}.png")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved: {path}")

    return {
        "shap_values":   shap_dict,
        "importance_df": importance_df,
        "top_drivers":   drivers,
        "figures":       figures,
    }
