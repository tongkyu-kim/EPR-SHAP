"""
Stage 1: Average Treatment Effect (ATE) estimation for the EPR experiment.

Implements
----------
1. Difference-in-means (DiM) Welch t-tests — each arm vs. control
2. ANCOVA: OLS with HC2 robust standard errors (covariate-adjusted ATE)
3. Joint F-test: any treatment effect across all three arms?
4. Cohen's d effect sizes
5. Summary table for export (LaTeX / CSV)

Variable names follow the codebook (survey/codebook/codebook.xlsx).

Outcomes
--------
Primary   : BAE_EPR  — budget share allocated to EPR compliance (%)
            CI_MEAN  — EPR compliance intentions composite (1–7)
Secondary : PP_MEAN  — policy preferences composite (1–7)

Treatment arms (TREAT)
----------------------
0 = Control
1 = Regulatory pressure
2 = Reputational pressure
3 = Market pressure

References
----------
Lin (2013). Agnostic regression adjustments to experimental data.
  https://arxiv.org/abs/1208.2301
Imbens & Rubin (2015). Causal Inference for Statistics, Ch. 7.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from typing import Dict, Optional

from data.synthetic.generate_epr_data import COVARIATE_COLS, TREATMENT_LABELS


ARMS = {1: "Regulatory", 2: "Reputational", 3: "Market"}

OUTCOMES = {
    "BAE_EPR":  "Budget Allocation: EPR Compliance (%)",
    "CI_MEAN":  "EPR Compliance Intentions (1–7)",
    "PP_MEAN":  "Policy Preferences (1–7)",
}

DEFAULT_COVARIATES = COVARIATE_COLS


# ---------------------------------------------------------------------------
# Difference-in-means
# ---------------------------------------------------------------------------

def dim_tests(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """
    Welch t-test: each treatment arm vs. control for a single outcome.

    Returns one row per arm with:
        arm, n_treat, n_control, mean_treat, mean_control,
        diff, t_stat, p_value, ci_lower, ci_upper, cohens_d
    """
    control = df[df["TREAT"] == 0][outcome].dropna()
    records = []

    for arm, label in ARMS.items():
        treat = df[df["TREAT"] == arm][outcome].dropna()

        t_stat, p_val = stats.ttest_ind(treat, control, equal_var=False)
        diff = treat.mean() - control.mean()

        # 95% CI via Welch degrees of freedom
        se = np.sqrt(treat.var() / len(treat) + control.var() / len(control))
        df_w = (treat.var() / len(treat) + control.var() / len(control)) ** 2 / (
            (treat.var() / len(treat)) ** 2 / (len(treat) - 1)
            + (control.var() / len(control)) ** 2 / (len(control) - 1)
        )
        t_crit = stats.t.ppf(0.975, df_w)

        # Cohen's d (pooled SD)
        pooled_sd = np.sqrt(
            (treat.var() * (len(treat) - 1) + control.var() * (len(control) - 1))
            / (len(treat) + len(control) - 2)
        )
        cohens_d = diff / pooled_sd if pooled_sd > 0 else np.nan

        records.append({
            "arm":        label,
            "n_treat":    len(treat),
            "n_control":  len(control),
            "mean_treat": round(treat.mean(), 4),
            "mean_ctrl":  round(control.mean(), 4),
            "diff":       round(diff, 4),
            "t_stat":     round(t_stat, 4),
            "p_value":    round(p_val, 4),
            "ci_lower":   round(diff - t_crit * se, 4),
            "ci_upper":   round(diff + t_crit * se, 4),
            "cohens_d":   round(cohens_d, 4),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# ANCOVA: covariate-adjusted ATE with HC2 robust SEs
# ---------------------------------------------------------------------------

def ancova_ate(
    df: pd.DataFrame,
    outcome: str,
    covariates: Optional[list] = None,
    cov_type: str = "HC2",
) -> pd.DataFrame:
    """
    Covariate-adjusted ATE via OLS with HC2 robust standard errors.

    Model: Y = α + β₁·T_REG + β₂·T_REP + β₃·T_MKT + γ'X + ε

    HC2 down-weights high-leverage observations and is the recommended
    covariance estimator for randomised experiments (Lin 2013).
    """
    covariates = covariates or DEFAULT_COVARIATES
    avail_covs = [c for c in covariates if c in df.columns]

    df_m = df[df["TREAT"].isin([0, 1, 2, 3])].copy()
    for arm, label in ARMS.items():
        df_m[f"T_{label.upper()}"] = (df_m["TREAT"] == arm).astype(int)

    dummy_cols = [f"T_{label.upper()}" for label in ARMS.values()]
    X = sm.add_constant(df_m[dummy_cols + avail_covs])
    y = df_m[outcome]

    mask = X.notna().all(axis=1) & y.notna()
    result = sm.OLS(y[mask], X[mask]).fit(cov_type=cov_type)

    # Ensure params are in a named Series (statsmodels 0.14 compatibility)
    param_names = list(result.model.exog_names)
    params  = pd.Series(result.params,  index=param_names)
    bses    = pd.Series(result.bse,     index=param_names)
    tvals   = pd.Series(result.tvalues, index=param_names)
    pvals   = pd.Series(result.pvalues, index=param_names)
    ci      = pd.DataFrame(result.conf_int(), index=param_names)

    records = []
    for arm, label in ARMS.items():
        name = f"T_{label.upper()}"
        records.append({
            "arm":       label,
            "coef":      round(float(params[name]),   4),
            "se":        round(float(bses[name]),     4),
            "t_stat":    round(float(tvals[name]),    4),
            "p_value":   round(float(pvals[name]),    4),
            "ci_lower":  round(float(ci.loc[name, 0]), 4),
            "ci_upper":  round(float(ci.loc[name, 1]), 4),
            "n_obs":     int(mask.sum()),
            "r_squared": round(float(result.rsquared), 4),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Joint F-test
# ---------------------------------------------------------------------------

def joint_f_test(
    df: pd.DataFrame,
    outcome: str,
    covariates: Optional[list] = None,
    cov_type: str = "HC2",
) -> Dict:
    """
    Wald F-test of H₀: β_REG = β_REP = β_MKT = 0 jointly.
    """
    covariates = covariates or DEFAULT_COVARIATES
    avail_covs = [c for c in covariates if c in df.columns]
    df_m = df[df["TREAT"].isin([0, 1, 2, 3])].copy()

    for arm, label in ARMS.items():
        df_m[f"T_{label.upper()}"] = (df_m["TREAT"] == arm).astype(int)

    dummy_cols = [f"T_{label.upper()}" for label in ARMS.values()]
    X = sm.add_constant(df_m[dummy_cols + avail_covs])
    y = df_m[outcome]
    mask = X.notna().all(axis=1) & y.notna()

    full_model = sm.OLS(y[mask], X[mask]).fit(cov_type=cov_type)
    hypotheses  = [f"T_{label.upper()} = 0" for label in ARMS.values()]
    f_test = full_model.f_test(hypotheses)

    return {
        "f_stat":     round(float(f_test.fvalue), 4),
        "p_value":    round(float(f_test.pvalue), 4),
        "df_num":     int(f_test.df_num),
        "df_denom":   int(f_test.df_denom),
        "any_effect": float(f_test.pvalue) < 0.05,
    }


# ---------------------------------------------------------------------------
# Run all outcomes
# ---------------------------------------------------------------------------

def estimate_all_ates(
    df: pd.DataFrame,
    covariates: Optional[list] = None,
    cov_type: str = "HC2",
    include_dim: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Estimate ANCOVA ATEs for every outcome variable.

    Returns a dict: {outcome_col: DataFrame of arm-level results}.
    """
    results = {}

    print("=" * 72)
    print("STAGE 1: AVERAGE TREATMENT EFFECTS  (ANCOVA + HC2 robust SEs)")
    print("=" * 72)

    for outcome, label in OUTCOMES.items():
        if outcome not in df.columns:
            continue

        ate_df  = ancova_ate(df, outcome, covariates, cov_type)
        f_info  = joint_f_test(df, outcome, covariates, cov_type)
        ate_df["outcome"] = label
        results[outcome] = ate_df

        print(f"\n{'─'*40}")
        print(f"Outcome: {label}")
        print(ate_df[["arm", "coef", "se", "p_value",
                       "ci_lower", "ci_upper"]].to_string(index=False))
        stars = ("***" if f_info["p_value"] < 0.001 else
                 "**"  if f_info["p_value"] < 0.01  else
                 "*"   if f_info["p_value"] < 0.05  else "")
        print(f"  Joint F({f_info['df_num']},{f_info['df_denom']}) = "
              f"{f_info['f_stat']},  p = {f_info['p_value']} {stars}")

        if include_dim:
            dim_df = dim_tests(df, outcome)
            print("  DiM (unadjusted):")
            print("  " + dim_df[["arm", "diff", "t_stat",
                                  "p_value", "cohens_d"]].to_string(index=False))

    return results


def ate_summary_table(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine all outcomes into one wide-format table for export."""
    rows = []
    for outcome, df_res in results.items():
        for _, row in df_res.iterrows():
            rows.append({
                "Outcome":  row.get("outcome", outcome),
                "Arm":      row["arm"],
                "ATE":      row["coef"],
                "SE":       row["se"],
                "p":        row["p_value"],
                "95% CI":   f"[{row['ci_lower']}, {row['ci_upper']}]",
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.preprocessing.data_cleaning import load_synthetic

    df, _ = load_synthetic(n=800)
    results = estimate_all_ates(df)
    tbl = ate_summary_table(results)
    print("\n\nSUMMARY TABLE:")
    print(tbl.to_string(index=False))
    tbl.to_csv("outputs/tables/ate_results.csv", index=False)
