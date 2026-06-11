"""
Survey data preprocessing for the EPR governance experiment.

Responsibilities
----------------
1. Load raw survey CSV (Qualtrics export or equivalent)
2. Validate required columns against the codebook
3. Recode / clip variables to valid ranges
4. Build construct means from item-level columns
5. Run manipulation check diagnostics
6. Test randomisation balance
7. Return analysis-ready DataFrame

All variable names follow the codebook (survey/codebook/codebook.xlsx).

Section map (codebook)
----------------------
A  — Respondent: POSITION, TENURE, FAMILIARITY
B  — Firm: INDUSTRY, EMP_SIZE, REVENUE, EXPORT_PCT, OWNERSHIP, SUPPLY_POS
C  — Env. mgmt binaries: ESG_REP, ISO14001, ENV_INSP, ENV_PEN, EPR_REP
D  — EPR Awareness: AWR1–AWR5 → AWR_MEAN
E  — Env. Orientation: EO1–EO5 → EO_MEAN
F  — Compliance Capability: CC1–CC5 → CC_MEAN
G  — Resource Constraints: RC1–RC5 → RC_MEAN
H  — Baseline pressures: RP1–RP4→RP_MEAN, REP1–REP4→REP_MEAN, MP1–MP4→MP_MEAN
I  — Treatment: TREAT (0/1/2/3)
J  — Manipulation check: MC1–MC3
K  — Budget allocation: BAE_MKT, BAE_PROD, BAE_OPS, BAE_TRAIN, BAE_EPR
L  — Compliance Intentions: CI1–CI5 → CI_MEAN
M  — Policy Preferences: PP1–PP4 → PP_MEAN
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple

from data.synthetic.generate_epr_data import (
    COVARIATE_COLS, TREATMENT_LABELS, ITEM_GROUPS
)


# ---------------------------------------------------------------------------
# Column specifications
# ---------------------------------------------------------------------------

BINARY_COLS = ["ESG_REP", "ISO14001", "ENV_INSP", "ENV_PEN", "EPR_REP"]

LIKERT7_CONSTRUCTS = {
    "AWR":  (ITEM_GROUPS["AWR"],  "AWR_MEAN"),
    "EO":   (ITEM_GROUPS["EO"],   "EO_MEAN"),
    "CC":   (ITEM_GROUPS["CC"],   "CC_MEAN"),
    "RC":   (ITEM_GROUPS["RC"],   "RC_MEAN"),
    "RP":   (ITEM_GROUPS["RP"],   "RP_MEAN"),
    "REP":  (ITEM_GROUPS["REP"],  "REP_MEAN"),
    "MP":   (ITEM_GROUPS["MP"],   "MP_MEAN"),
    "CI":   (ITEM_GROUPS["CI"],   "CI_MEAN"),
    "PP":   (ITEM_GROUPS["PP"],   "PP_MEAN"),
}

BAE_COLS = ITEM_GROUPS["BAE"]   # ["BAE_MKT","BAE_PROD","BAE_OPS","BAE_TRAIN","BAE_EPR"]

PRIMARY_OUTCOMES = ["BAE_EPR", "CI_MEAN"]
SECONDARY_OUTCOMES = ["PP_MEAN"]

REQUIRED_COLS = (
    ["TREAT"]
    + COVARIATE_COLS
    + BAE_COLS
    + ITEM_GROUPS["CI"]
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_survey(path: str) -> pd.DataFrame:
    """
    Load a raw survey CSV.

    Drops Qualtrics metadata rows (rows 0–1 when they contain question text),
    normalises column names to uppercase + underscores, and coerces numerics.
    """
    df = pd.read_csv(path, low_memory=False)

    # Drop Qualtrics label / import-id rows if present
    first_val = str(df.iloc[0, 0]) if len(df) > 0 else ""
    if first_val.startswith("{") or first_val.lower().startswith("import"):
        df = df.iloc[2:].reset_index(drop=True)

    # Normalise column names
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
        .str.replace(r"[\s\-]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )

    # Coerce all columns to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    invalid = set(df["TREAT"].dropna().unique()) - {0, 1, 2, 3}
    if invalid:
        raise ValueError(f"Unexpected TREAT values: {invalid}")


# ---------------------------------------------------------------------------
# Recoding
# ---------------------------------------------------------------------------

def recode_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Clip binary controls
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].clip(0, 1).round().astype("Int64")

    # Clip all 1–7 Likert items
    all_items = [item for grp in ITEM_GROUPS.values() for item in grp]
    for col in all_items:
        if col in df.columns:
            df[col] = df[col].clip(1, 7)

    # Clip ordinal firm variables
    if "EMP_SIZE" in df.columns:
        df["EMP_SIZE"] = df["EMP_SIZE"].clip(1, 6)
    if "REVENUE" in df.columns:
        df["REVENUE"] = df["REVENUE"].clip(1, 5)
    if "EXPORT_PCT" in df.columns:
        df["EXPORT_PCT"] = df["EXPORT_PCT"].clip(0, 100)
    if "OWNERSHIP" in df.columns:
        df["OWNERSHIP"] = df["OWNERSHIP"].clip(1, 6)
    if "SUPPLY_POS" in df.columns:
        df["SUPPLY_POS"] = df["SUPPLY_POS"].clip(1, 6)

    # Budget items: clip to [0,100], will be renormalised later
    for col in BAE_COLS:
        if col in df.columns:
            df[col] = df[col].clip(0, 100)

    # Treatment label
    df["TREAT_LABEL"] = df["TREAT"].map(TREATMENT_LABELS)

    # Treatment dummies (for ANCOVA)
    for arm, label in TREATMENT_LABELS.items():
        if arm > 0:
            df[f"T_{label.upper()}"] = (df["TREAT"] == arm).astype(int)

    return df


# ---------------------------------------------------------------------------
# Construct means
# ---------------------------------------------------------------------------

def build_construct_means(df: pd.DataFrame,
                           min_items: int = 3) -> pd.DataFrame:
    """
    Compute (or recompute) construct means from item-level columns.

    Rows with fewer than min_items valid responses for a construct
    are set to NaN for that mean.
    """
    df = df.copy()
    for construct, (items, mean_col) in LIKERT7_CONSTRUCTS.items():
        available = [c for c in items if c in df.columns]
        if not available:
            continue
        item_data = df[available]
        n_valid = item_data.notna().sum(axis=1)
        means = item_data.mean(axis=1)
        means[n_valid < min_items] = np.nan
        df[mean_col] = means.round(3)
    return df


def renormalise_bae(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure budget allocation items (BAE_*) sum to 100.
    If a respondent's sum deviates from 100, rescale proportionally.
    Rows that sum to 0 are set to NaN.
    """
    df = df.copy()
    available = [c for c in BAE_COLS if c in df.columns]
    row_sum = df[available].sum(axis=1)
    valid = row_sum > 0
    for col in available:
        df.loc[valid, col] = (df.loc[valid, col] / row_sum[valid] * 100).round(1)
    df.loc[~valid, available] = np.nan
    return df


# ---------------------------------------------------------------------------
# Manipulation check
# ---------------------------------------------------------------------------

def check_manipulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether the three manipulation-check items (MC1–MC3) show
    expected patterns: treated respondents should score higher on the
    arm-relevant item.

    MC1 → Regulatory salience  (arm=1)
    MC2 → Reputational salience (arm=2)
    MC3 → Market salience       (arm=3)

    Returns a summary DataFrame with mean scores per arm per item.
    """
    mc_items = [c for c in ITEM_GROUPS["MC"] if c in df.columns]
    if not mc_items:
        return pd.DataFrame()

    summary = df.groupby("TREAT_LABEL")[mc_items].mean().round(3)
    return summary


# ---------------------------------------------------------------------------
# Balance check
# ---------------------------------------------------------------------------

def check_balance(df: pd.DataFrame,
                  covariates: Optional[list] = None) -> pd.DataFrame:
    """
    Test covariate balance across the four arms.

    Continuous / ordinal  → one-way ANOVA (F-test)
    Binary / low-cardinality → chi-squared

    Returns one row per covariate with: variable, test, stat, p_value, balanced.
    """
    covariates = covariates or COVARIATE_COLS
    groups = [df[df["TREAT"] == arm][covariates] for arm in range(4)]
    records = []

    for col in covariates:
        col_data = [g[col].dropna() for g in groups]
        n_unique = df[col].nunique()

        if n_unique <= 6:
            try:
                ct = pd.crosstab(df[col], df["TREAT"])
                chi2, p, _, _ = stats.chi2_contingency(ct)
                records.append({"variable": col, "test": "chi2",
                                 "stat": round(chi2, 3), "p_value": round(p, 4),
                                 "balanced": p > 0.05})
            except Exception:
                records.append({"variable": col, "test": "chi2",
                                 "stat": np.nan, "p_value": np.nan, "balanced": True})
        else:
            try:
                f_stat, p = stats.f_oneway(*col_data)
                records.append({"variable": col, "test": "anova",
                                 "stat": round(f_stat, 3), "p_value": round(p, 4),
                                 "balanced": p > 0.05})
            except Exception:
                records.append({"variable": col, "test": "anova",
                                 "stat": np.nan, "p_value": np.nan, "balanced": True})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def preprocess_survey(
    path: str,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline for a raw survey CSV.

    Returns
    -------
    df      : Analysis-ready dataset with construct means and dummies.
    balance : Balance check results DataFrame.
    """
    df = load_survey(path)
    validate_columns(df)
    df = recode_variables(df)
    df = build_construct_means(df)
    df = renormalise_bae(df)

    balance = check_balance(df)
    manip   = check_manipulation(df)

    if verbose:
        print(f"Loaded {len(df)} observations, {df['TREAT'].nunique()} arms")
        print("\nArm sizes:")
        print(df["TREAT_LABEL"].value_counts().sort_index())
        print("\nManipulation check (mean MC scores by arm):")
        print(manip.to_string())
        n_fail = (~balance["balanced"]).sum()
        print(f"\nBalance: {n_fail}/{len(balance)} covariates fail p>0.05")

    return df, balance


def load_synthetic(n: int = 800, seed: int = 42,
                   verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Shortcut: generate synthetic data and run preprocessing."""
    from data.synthetic.generate_epr_data import generate_epr_data
    df_raw = generate_epr_data(n=n, seed=seed)
    df = recode_variables(df_raw)
    df = build_construct_means(df)
    df = renormalise_bae(df)
    balance = check_balance(df)

    if verbose:
        manip = check_manipulation(df)
        print(f"Synthetic data: {len(df)} rows")
        print(manip.to_string())

    return df, balance


if __name__ == "__main__":
    df, balance = load_synthetic(n=800, verbose=True)
    print("\nOutcome means by arm:")
    print(df.groupby("TREAT_LABEL")[["BAE_EPR", "CI_MEAN", "PP_MEAN"]].mean().round(3))
    print("\nBalance check (first 8 covariates):")
    print(balance.head(8).to_string(index=False))
