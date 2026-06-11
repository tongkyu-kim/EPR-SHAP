"""
Stage 2: Heterogeneous Treatment Effect (CATE) estimation for the EPR experiment.

Strategy
--------
EconML's CausalForestDML supports two modes for a 4-arm experiment:

  A) Pairwise binary models (recommended)
     One CausalForestDML per arm, each arm vs. control only.
     → Cleanest per-mechanism SHAP interpretation.
     → Direct correspondence to ATE stage comparisons.

  B) Discrete multi-arm model
     Single model with discrete_treatment=True and a RandomForestClassifier
     for the treatment nuisance model.
     → Joint estimation; useful for cross-arm CATE comparisons.

Theory-motivated heterogeneity hypotheses (concept note §8):
  H1: Export-oriented firms (high EXPORT_PCT) → stronger Market response
  H2: Previously inspected firms (ENV_INSP=1) → stronger Regulatory response
  H3: Environmentally engaged firms (high EO_MEAN) → stronger Reputational response

Variable names follow codebook (survey/codebook/codebook.xlsx).
"""

import numpy as np
import pandas as pd
from scipy import stats as sc_stats
from typing import Dict, Optional, Tuple

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from econml.dml import CausalForestDML

from data.synthetic.generate_epr_data import COVARIATE_COLS, TREATMENT_LABELS


ARMS = {1: "Regulatory", 2: "Reputational", 3: "Market"}
DEFAULT_COVARIATES = COVARIATE_COLS
DEFAULT_OUTCOME    = "BAE_EPR"


# ---------------------------------------------------------------------------
# Mode A: Pairwise binary CausalForestDML (recommended)
# ---------------------------------------------------------------------------

def fit_pairwise_causal_forests(
    df: pd.DataFrame,
    outcome: str = DEFAULT_OUTCOME,
    covariates: Optional[list] = None,
    n_estimators: int = 500,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, CausalForestDML]:
    """
    Fit one CausalForestDML per treatment arm (each vs. control only).

    Parameters
    ----------
    df           : Analysis-ready DataFrame from preprocess_survey / load_synthetic.
    outcome      : "BAE_EPR" (primary, default) or "CI_MEAN" / "PP_MEAN".
    covariates   : Feature columns. Defaults to COVARIATE_COLS.
    n_estimators : Trees per causal forest.
    random_state : Seed.

    Returns
    -------
    dict  {arm_label: fitted CausalForestDML}
    """
    covariates = covariates or DEFAULT_COVARIATES
    X_cols = [c for c in covariates if c in df.columns]
    models = {}

    for arm, label in ARMS.items():
        sub = df[df["TREAT"].isin([0, arm])].copy()
        sub["_T"] = (sub["TREAT"] == arm).astype(int)

        X = sub[X_cols].fillna(sub[X_cols].median()).values.astype(float)
        T = sub["_T"].values
        Y = sub[outcome].values

        model_y = GradientBoostingRegressor(
            n_estimators=100, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            random_state=random_state,
        )
        model_t = GradientBoostingRegressor(
            n_estimators=100, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            random_state=random_state,
        )

        est = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_leaf=10,
            max_samples=0.5,
            honest=True,
            cv=5,
            random_state=random_state,
            verbose=0,
        )
        est.fit(Y, T, X=X)
        models[label] = est

        if verbose:
            ate      = est.ate(X)
            ate_se   = est.ate_inference(X).stderr_mean
            print(f"  [{label}] ATE = {ate:.4f}  (SE = {ate_se:.4f})")

    return models


# ---------------------------------------------------------------------------
# Mode B: Discrete multi-arm CausalForestDML
# ---------------------------------------------------------------------------

def fit_multiarm_causal_forest(
    df: pd.DataFrame,
    outcome: str = DEFAULT_OUTCOME,
    covariates: Optional[list] = None,
    n_estimators: int = 500,
    random_state: int = 42,
    verbose: bool = True,
) -> CausalForestDML:
    """
    Single CausalForestDML with discrete_treatment=True.

    model.effect(X) returns shape (n, 3): one CATE column per non-control arm.
    """
    covariates = covariates or DEFAULT_COVARIATES
    X_cols = [c for c in covariates if c in df.columns]

    X = df[X_cols].fillna(df[X_cols].median()).values.astype(float)
    T = df["TREAT"].values
    Y = df[outcome].values

    mask = ~np.isnan(Y)
    X, T, Y = X[mask], T[mask], Y[mask]

    est = CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=5,
            random_state=random_state,
        ),
        model_t=RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=5,
            random_state=random_state,
        ),
        discrete_treatment=True,
        categories=[0, 1, 2, 3],
        n_estimators=n_estimators,
        min_samples_leaf=10,
        honest=True,
        cv=5,
        random_state=random_state,
        verbose=0,
    )
    est.fit(Y, T, X=X)

    if verbose:
        effects = est.effect(X)
        for i, label in enumerate(ARMS.values()):
            print(f"  [{label}] mean CATE = {effects[:, i].mean():.4f}")

    return est


# ---------------------------------------------------------------------------
# CATE extraction
# ---------------------------------------------------------------------------

def extract_cates(
    models: Dict[str, CausalForestDML],
    df: pd.DataFrame,
    covariates: Optional[list] = None,
) -> pd.DataFrame:
    """
    Predict CATEs for all firms and attach to the DataFrame.

    Adds columns:
        CATE_REGULATORY, CATE_REGULATORY_SE
        CATE_REPUTATIONAL, CATE_REPUTATIONAL_SE
        CATE_MARKET, CATE_MARKET_SE
    """
    covariates = covariates or DEFAULT_COVARIATES
    X_cols = [c for c in covariates if c in df.columns]
    X = df[X_cols].fillna(df[X_cols].median()).values.astype(float)
    df_out = df.copy()

    for label, model in models.items():
        col    = f"CATE_{label.upper()}"
        se_col = f"CATE_{label.upper()}_SE"
        df_out[col]    = model.effect(X)
        df_out[se_col] = model.effect_inference(X).stderr

    return df_out


# ---------------------------------------------------------------------------
# Heterogeneity hypothesis tests
# ---------------------------------------------------------------------------

def test_heterogeneity_hypotheses(df_cate: pd.DataFrame) -> pd.DataFrame:
    """
    Test the three theory-motivated heterogeneity hypotheses via
    Welch t-tests on high vs. low moderator subgroups.

    H1: EXPORT_PCT (high) → stronger Market CATE
    H2: ENV_INSP = 1      → stronger Regulatory CATE
    H3: EO_MEAN (high)    → stronger Reputational CATE
    """
    hypotheses = [
        # (cate_col, moderator condition, hypothesis label)
        ("CATE_REGULATORY",
         df_cate["ENV_INSP"] == 1,
         "H2: ENV_INSP=1 → stronger Regulatory response"),

        ("CATE_REPUTATIONAL",
         df_cate["EO_MEAN"] >= df_cate["EO_MEAN"].median(),
         "H3: High EO_MEAN → stronger Reputational response"),

        ("CATE_MARKET",
         df_cate["EXPORT_PCT"] >= df_cate["EXPORT_PCT"].median(),
         "H1: High EXPORT_PCT → stronger Market response"),
    ]

    records = []
    for cate_col, high_mask, label in hypotheses:
        if cate_col not in df_cate.columns:
            continue
        high = df_cate[high_mask][cate_col].dropna()
        low  = df_cate[~high_mask][cate_col].dropna()

        t, p = sc_stats.ttest_ind(high, low, equal_var=False)
        records.append({
            "Hypothesis":       label,
            "CATE (high mod.)": round(high.mean(), 4),
            "CATE (low mod.)":  round(low.mean(), 4),
            "Difference":       round(high.mean() - low.mean(), 4),
            "t_stat":           round(float(t), 4),
            "p_value":          round(float(p), 4),
            "n_high": len(high),
            "n_low":  len(low),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_cate_distributions(df_cate: pd.DataFrame, figsize=(14, 4)):
    """KDE plots of CATE distributions for all three arms."""
    import matplotlib.pyplot as plt

    cols = {
        "CATE_REGULATORY":   ("#e74c3c", "Regulatory Pressure"),
        "CATE_REPUTATIONAL": ("#3498db", "Reputational Pressure"),
        "CATE_MARKET":       ("#2ecc71", "Market Pressure"),
    }
    available = {lab: df_cate[col].dropna()
                 for col, (_, lab) in cols.items() if col in df_cate.columns}

    fig, axes = plt.subplots(1, len(available), figsize=figsize, sharey=False)
    if len(available) == 1:
        axes = [axes]

    for ax, (col, (color, lab)) in zip(
        axes, {k: v for k, v in cols.items() if k in df_cate.columns}.items()
    ):
        ser = df_cate[col].dropna()
        ser.plot.kde(ax=ax, color=color, linewidth=2)
        ax.axvline(ser.mean(),   linestyle="--", color="black", alpha=0.6, label="Mean")
        ax.axvline(0,            linestyle=":",  color="red",   alpha=0.4)
        ax.set_title(f"{lab}\nCATE Distribution")
        ax.set_xlabel("Estimated Treatment Effect (BAE_EPR, %)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Heterogeneous Treatment Effects by Governance Mechanism",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    return fig


def plot_cate_by_subgroup(
    df_cate: pd.DataFrame,
    cate_col: str,
    group_col: str,
    n_quantiles: int = 5,
    figsize=(8, 5),
):
    """
    Bar chart of mean CATE across quantile groups of a continuous moderator.
    For binary/categorical moderators, groups are used directly.
    """
    import matplotlib.pyplot as plt

    df_plot = df_cate[[cate_col, group_col]].dropna().copy()
    n_unique = df_plot[group_col].nunique()

    if n_unique > 5:
        df_plot["group"] = pd.qcut(
            df_plot[group_col], n_quantiles,
            labels=[f"Q{i+1}" for i in range(n_quantiles)],
            duplicates="drop",
        )
    else:
        df_plot["group"] = df_plot[group_col].astype(str)

    agg = df_plot.groupby("group")[cate_col].agg(
        mean="mean",
        sem=lambda x: x.std() / np.sqrt(len(x)),
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(agg)), agg["mean"],
           yerr=1.96 * agg["sem"], capsize=5,
           color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg.index, rotation=30, ha="right")
    ax.axhline(0, linestyle="--", color="red", alpha=0.5)
    ax.set_title(f"Mean CATE by {group_col}  ({cate_col})")
    ax.set_ylabel("Mean CATE (±95% CI)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_cate_pipeline(
    df: pd.DataFrame,
    outcome: str = DEFAULT_OUTCOME,
    covariates: Optional[list] = None,
    mode: str = "pairwise",
    n_estimators: int = 500,
    random_state: int = 42,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end CATE estimation.

    Parameters
    ----------
    mode : "pairwise" (recommended) or "multiarm"

    Returns
    -------
    models      : dict of fitted models
    df_cate     : df augmented with CATE_* columns
    hypotheses  : heterogeneity hypothesis test results
    """
    print("=" * 72)
    print(f"STAGE 2: CATE ESTIMATION  --  mode={mode.upper()}"
          f",  outcome={outcome}")
    print("=" * 72)

    covariates = covariates or DEFAULT_COVARIATES

    if mode == "pairwise":
        models  = fit_pairwise_causal_forests(
            df, outcome, covariates, n_estimators, random_state
        )
        df_cate = extract_cates(models, df, covariates)
    else:
        model_ma = fit_multiarm_causal_forest(
            df, outcome, covariates, n_estimators, random_state
        )
        models = {"multiarm": model_ma}
        X_cols = [c for c in covariates if c in df.columns]
        X = df[X_cols].fillna(df[X_cols].median()).values.astype(float)
        effects = model_ma.effect(X)
        df_cate = df.copy()
        for i, label in enumerate(ARMS.values()):
            df_cate[f"CATE_{label.upper()}"] = effects[:, i]

    print("\nHeterogeneity Hypothesis Tests:")
    hypotheses = test_heterogeneity_hypotheses(df_cate)
    print(hypotheses.to_string(index=False))

    return models, df_cate, hypotheses


if __name__ == "__main__":
    from src.preprocessing.data_cleaning import load_synthetic

    df, _ = load_synthetic(n=800)
    models, df_cate, hyp = run_cate_pipeline(df, mode="pairwise",
                                              n_estimators=200)
    fig = plot_cate_distributions(df_cate)
    fig.savefig("outputs/figures/cate_distributions.png",
                dpi=150, bbox_inches="tight")
    print("Saved CATE distribution plot.")
