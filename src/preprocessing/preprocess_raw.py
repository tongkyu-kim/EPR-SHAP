"""
Preprocessing script for the raw survey test data (Excel → clean CSV).

Reads:  data/raw/survey_data_test.xlsx
Writes: data/processed/survey_data_clean.csv
        data/processed/balance_check.csv
        data/processed/manipulation_check.csv
        data/processed/preprocessing_report.txt

What this script does
---------------------
1. Load the Excel file with openpyxl (avoids pandas Excel dependency issues)
2. Round floating-point artefacts in pre-computed _MEAN columns
3. Recompute all construct means from raw items (authoritative; replaces pre-computed)
4. Validate value ranges and BAE sum constraint
5. Add convenience columns (treatment dummies, CI_STD, CI_ZSCORED)
6. Run and save balance check + manipulation check diagnostics
7. Export clean dataset

Run
---
    python -m src.preprocessing.preprocess_raw
    # or from repo root:
    python src/preprocessing/preprocess_raw.py
"""

import os
import sys
import math
import warnings
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import stats

# Allow running from repo root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data.synthetic.generate_epr_data import (
    COVARIATE_COLS, TREATMENT_LABELS, ITEM_GROUPS
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_PATH       = ROOT / "data" / "raw" / "survey_data_test.xlsx"
PROCESSED_DIR  = ROOT / "data" / "processed"
CLEAN_PATH     = PROCESSED_DIR / "survey_data_clean.csv"
BALANCE_PATH   = PROCESSED_DIR / "balance_check.csv"
MANIP_PATH     = PROCESSED_DIR / "manipulation_check.csv"
REPORT_PATH    = PROCESSED_DIR / "preprocessing_report.txt"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants (from codebook)
# ---------------------------------------------------------------------------

BINARY_COLS = ["ESG_REP", "ISO14001", "ENV_INSP", "ENV_PEN", "EPR_REP"]
BAE_COLS    = ITEM_GROUPS["BAE"]

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

ORDINAL_RANGES = {
    "EMP_SIZE":  (1, 6),
    "REVENUE":   (1, 5),
    "OWNERSHIP": (1, 6),
    "SUPPLY_POS":(1, 6),
    "INDUSTRY":  (1, 7),
    "POSITION":  (1, 7),
}


# ---------------------------------------------------------------------------
# Step 1: Load
# ---------------------------------------------------------------------------

def load_excel(path: Path) -> pd.DataFrame:
    """Load xlsx with openpyxl engine; coerce all numeric columns."""
    df = pd.read_excel(path, engine="openpyxl")
    # Strip whitespace from column names
    df.columns = df.columns.str.strip().str.upper().str.replace(r"[\s\-]+", "_", regex=True)
    print(f"  Loaded: {len(df)} rows × {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Step 2: Validate structure
# ---------------------------------------------------------------------------

def validate_structure(df: pd.DataFrame, report: list) -> None:
    required = ["TREAT"] + COVARIATE_COLS + BAE_COLS + ITEM_GROUPS["CI"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    invalid_treat = set(df["TREAT"].dropna().unique()) - {0, 1, 2, 3}
    if invalid_treat:
        raise ValueError(f"Unexpected TREAT values: {invalid_treat}")

    report.append("✓ All required columns present")
    report.append(f"✓ TREAT values: {sorted(df['TREAT'].unique().tolist())}")


# ---------------------------------------------------------------------------
# Step 3: Clip and validate ranges
# ---------------------------------------------------------------------------

def clean_ranges(df: pd.DataFrame, report: list) -> pd.DataFrame:
    df = df.copy()
    issues = []

    # Binary variables
    for col in BINARY_COLS:
        if col in df.columns:
            bad = ((df[col] != 0) & (df[col] != 1)).sum()
            if bad:
                issues.append(f"  {col}: {bad} out-of-range values clipped to 0/1")
            df[col] = df[col].clip(0, 1).round().astype("Int64")

    # All Likert-7 items — explicitly exclude BAE (0–100%) and MC (1–7 but separate)
    likert_groups = ["AWR", "EO", "CC", "RC", "RP", "REP", "MP", "CI", "PP", "MC"]
    all_items = [it for grp_name in likert_groups for it in ITEM_GROUPS.get(grp_name, [])]
    for col in all_items:
        if col in df.columns:
            bad = ((df[col] < 1) | (df[col] > 7)).sum()
            if bad:
                issues.append(f"  {col}: {bad} out-of-range Likert values clipped to 1–7")
            df[col] = df[col].clip(1, 7)

    # Ordinal variables
    for col, (lo, hi) in ORDINAL_RANGES.items():
        if col in df.columns:
            bad = ((df[col] < lo) | (df[col] > hi)).sum()
            if bad:
                issues.append(f"  {col}: {bad} out-of-range values clipped to {lo}–{hi}")
            df[col] = df[col].clip(lo, hi)

    # EXPORT_PCT
    if "EXPORT_PCT" in df.columns:
        bad = ((df["EXPORT_PCT"] < 0) | (df["EXPORT_PCT"] > 100)).sum()
        if bad:
            issues.append(f"  EXPORT_PCT: {bad} out-of-range values clipped to 0–100")
        df["EXPORT_PCT"] = df["EXPORT_PCT"].clip(0, 100)

    if issues:
        report.append("⚠ Range issues found and corrected:")
        report.extend(issues)
    else:
        report.append("✓ All values within expected ranges")

    return df


# ---------------------------------------------------------------------------
# Step 4: Recompute construct means (authoritative; fixes fp artefacts)
# ---------------------------------------------------------------------------

def recompute_means(df: pd.DataFrame, report: list,
                    min_items: int = 3) -> pd.DataFrame:
    df = df.copy()
    for construct, (items, mean_col) in LIKERT7_CONSTRUCTS.items():
        avail = [c for c in items if c in df.columns]
        if not avail:
            report.append(f"⚠ {construct}: no item columns found, skipping mean")
            continue
        item_data = df[avail]
        n_valid   = item_data.notna().sum(axis=1)
        new_mean  = item_data.mean(axis=1).round(3)
        new_mean[n_valid < min_items] = np.nan

        if mean_col in df.columns:
            fp_fixed = (df[mean_col].round(3) != df[mean_col]).sum()
            if fp_fixed:
                report.append(f"  {mean_col}: rounded {fp_fixed} floating-point artefacts")
        df[mean_col] = new_mean

    report.append("✓ All construct means recomputed from items (min_items=3)")
    return df


# ---------------------------------------------------------------------------
# Step 5: Validate and renormalise BAE
# ---------------------------------------------------------------------------

def validate_bae(df: pd.DataFrame, report: list) -> pd.DataFrame:
    df   = df.copy()
    avail = [c for c in BAE_COLS if c in df.columns]
    # Ensure float dtype so rescaling can assign decimal values
    for col in avail:
        df[col] = df[col].astype(float)
    row_sum = df[avail].sum(axis=1)

    bad_sum = ((row_sum - 100).abs() > 1.5).sum()
    if bad_sum:
        report.append(f"⚠ BAE: {bad_sum} rows with sum ≠ 100 — rescaling")
        valid = row_sum > 0
        for col in avail:
            df.loc[valid, col] = (df.loc[valid, col] / row_sum[valid] * 100).round(1)
        df.loc[~valid, avail] = np.nan
    else:
        report.append("✓ BAE columns sum to 100 for all rows")

    report.append(f"  BAE_EPR range: {df['BAE_EPR'].min():.1f} – {df['BAE_EPR'].max():.1f}%"
                  f"  (mean={df['BAE_EPR'].mean():.2f}%)")
    return df


# ---------------------------------------------------------------------------
# Step 6: Add derived columns
# ---------------------------------------------------------------------------

def add_derived_columns(df: pd.DataFrame, report: list) -> pd.DataFrame:
    df = df.copy()

    # Treatment label (in case it's missing)
    df["TREAT_LABEL"] = df["TREAT"].map(TREATMENT_LABELS)

    # Treatment dummies for ANCOVA
    for arm, label in TREATMENT_LABELS.items():
        if arm > 0:
            df[f"T_{label.upper()}"] = (df["TREAT"] == arm).astype(int)

    # Standardised CI composite (z-score; useful for cross-scale comparisons)
    if "CI_MEAN" in df.columns:
        mu, sd = df["CI_MEAN"].mean(), df["CI_MEAN"].std()
        df["CI_ZSCORED"] = ((df["CI_MEAN"] - mu) / sd).round(4)

    report.append("✓ Added TREAT_LABEL, treatment dummies (T_*), CI_ZSCORED")
    return df


# ---------------------------------------------------------------------------
# Step 7: Missing-value report
# ---------------------------------------------------------------------------

def report_missing(df: pd.DataFrame, report: list) -> None:
    key_cols = (["TREAT", "BAE_EPR", "CI_MEAN", "PP_MEAN"]
                + COVARIATE_COLS[:8])
    total_missing = 0
    for col in key_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            total_missing += n
            if n:
                report.append(f"  Missing in {col}: {n} rows")

    if total_missing == 0:
        report.append("✓ No missing values in key columns")
    else:
        report.append(f"⚠ Total missing in key columns: {total_missing}")


# ---------------------------------------------------------------------------
# Step 8: Balance check
# ---------------------------------------------------------------------------

def run_balance_check(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in COVARIATE_COLS:
        if col not in df.columns:
            continue
        groups   = [df[df["TREAT"] == arm][col].dropna() for arm in range(4)]
        n_unique = df[col].nunique()

        try:
            if n_unique <= 6:
                ct = pd.crosstab(df[col], df["TREAT"])
                chi2, p, _, _ = stats.chi2_contingency(ct)
                records.append({"variable": col, "test": "chi2",
                                 "stat": round(chi2, 3), "p_value": round(p, 4),
                                 "balanced": p > 0.05})
            else:
                f_stat, p = stats.f_oneway(*groups)
                records.append({"variable": col, "test": "anova",
                                 "stat": round(f_stat, 3), "p_value": round(p, 4),
                                 "balanced": p > 0.05})
        except Exception as e:
            records.append({"variable": col, "test": "error",
                             "stat": None, "p_value": None, "balanced": None})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 9: Manipulation check
# ---------------------------------------------------------------------------

def run_manipulation_check(df: pd.DataFrame) -> pd.DataFrame:
    mc_items = [c for c in ITEM_GROUPS["MC"] if c in df.columns]
    if not mc_items:
        return pd.DataFrame()
    return df.groupby("TREAT_LABEL")[mc_items].mean().round(3)


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(verbose: bool = True) -> pd.DataFrame:
    report = []
    report.append("=" * 60)
    report.append("EPR SURVEY - PREPROCESSING REPORT")
    report.append("=" * 60)
    report.append(f"\nInput : {RAW_PATH}")
    report.append(f"Output: {CLEAN_PATH}")

    # --- Load ---
    report.append("\n[1] LOADING DATA")
    df = load_excel(RAW_PATH)
    report.append(f"  {len(df)} respondents × {len(df.columns)} columns")

    arm_counts = df["TREAT"].value_counts().sort_index()
    report.append("\n  Arm distribution:")
    for arm, n in arm_counts.items():
        report.append(f"    {TREATMENT_LABELS.get(int(arm), arm)}: n={n}")

    # --- Validate ---
    report.append("\n[2] STRUCTURAL VALIDATION")
    validate_structure(df, report)

    # --- Clean ---
    report.append("\n[3] RANGE CLEANING")
    df = clean_ranges(df, report)

    # --- Means ---
    report.append("\n[4] CONSTRUCT MEANS (recomputed from items)")
    df = recompute_means(df, report)

    # --- BAE ---
    report.append("\n[5] BUDGET ALLOCATION VALIDATION")
    df = validate_bae(df, report)

    # --- Missing ---
    report.append("\n[6] MISSING VALUES")
    report_missing(df, report)

    # --- Derived ---
    report.append("\n[7] DERIVED COLUMNS")
    df = add_derived_columns(df, report)

    # --- Balance ---
    report.append("\n[8] RANDOMISATION BALANCE CHECK")
    balance_df = run_balance_check(df)
    n_fail = (~balance_df["balanced"].fillna(True)).sum()
    report.append(f"  {n_fail}/{len(balance_df)} covariates fail p > 0.05 threshold")
    if n_fail > 0:
        fails = balance_df[~balance_df["balanced"]]["variable"].tolist()
        report.append(f"  ⚠ Imbalanced covariates: {fails}")
    else:
        report.append("  ✓ All covariates balanced across arms")

    # --- Manipulation ---
    report.append("\n[9] MANIPULATION CHECK")
    manip_df = run_manipulation_check(df)
    if not manip_df.empty:
        report.append("  MC1 (Regulatory salience), MC2 (Reputational), MC3 (Market)")
        report.append("  Mean scores by arm:")
        for arm_label, row in manip_df.iterrows():
            mc_vals = "  ".join([f"{col}={row[col]:.2f}" for col in manip_df.columns])
            report.append(f"    {arm_label:<14}: {mc_vals}")

    # --- Summary stats ---
    report.append("\n[10] OUTCOME SUMMARY BY ARM")
    outcomes = ["BAE_EPR", "CI_MEAN", "PP_MEAN"]
    avail_out = [o for o in outcomes if o in df.columns]
    summary = df.groupby("TREAT_LABEL")[avail_out].mean().round(3)
    report.append(summary.to_string())

    # --- Save ---
    df.to_csv(CLEAN_PATH, index=False)
    balance_df.to_csv(BALANCE_PATH, index=False)
    if not manip_df.empty:
        manip_df.to_csv(MANIP_PATH)

    report_text = "\n".join(report)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    if verbose:
        print(report_text)

    print(f"\n✓ Clean data saved  → {CLEAN_PATH}")
    print(f"✓ Balance check     → {BALANCE_PATH}")
    print(f"✓ Report saved      → {REPORT_PATH}")
    return df


# ---------------------------------------------------------------------------
# Readiness check — verifies the full pipeline can run
# ---------------------------------------------------------------------------

def readiness_check(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("STUDY READINESS CHECK")
    print("=" * 60)

    checks = []

    # Data checks
    checks.append(("Sample size (≥600 recommended)",
                   len(df) >= 600, f"n={len(df)}"))
    checks.append(("Balanced arms (150 per arm)",
                   all(df["TREAT"].value_counts() == 150),
                   df["TREAT"].value_counts().to_dict()))
    checks.append(("No missing in primary DV (BAE_EPR)",
                   df["BAE_EPR"].isna().sum() == 0,
                   f"missing={df['BAE_EPR'].isna().sum()}"))
    checks.append(("No missing in secondary DV (CI_MEAN)",
                   df["CI_MEAN"].isna().sum() == 0,
                   f"missing={df['CI_MEAN'].isna().sum()}"))

    # Covariate coverage
    cov_present = [c for c in COVARIATE_COLS if c in df.columns]
    checks.append((f"All {len(COVARIATE_COLS)} covariates present",
                   len(cov_present) == len(COVARIATE_COLS),
                   f"{len(cov_present)}/{len(COVARIATE_COLS)} found"))

    # Item-level columns
    all_items_present = all(
        item in df.columns
        for grp_items in ITEM_GROUPS.values()
        for item in grp_items
    )
    checks.append(("All item-level columns present (for CFA/reliability)",
                   all_items_present, ""))

    # Manipulation check columns
    mc_present = all(c in df.columns for c in ITEM_GROUPS["MC"])
    checks.append(("Manipulation check items (MC1–MC3) present",
                   mc_present, ""))

    # Code modules
    modules = {
        "src/preprocessing/data_cleaning.py":    ROOT / "src/preprocessing/data_cleaning.py",
        "src/ate/ate_estimation.py":              ROOT / "src/ate/ate_estimation.py",
        "src/cate/causal_forest_epr.py":          ROOT / "src/cate/causal_forest_epr.py",
        "src/shap_analysis/shap_epr.py":          ROOT / "src/shap_analysis/shap_epr.py",
        "src/utils/plot_utils.py":                ROOT / "src/utils/plot_utils.py",
        "data/synthetic/generate_epr_data.py":    ROOT / "data/synthetic/generate_epr_data.py",
    }
    for name, path in modules.items():
        checks.append((f"Module: {name}", path.exists(), ""))

    # Output directories
    for d in ["outputs/figures", "outputs/tables", "outputs/models"]:
        checks.append((f"Directory: {d}", (ROOT / d).exists(), ""))

    # Python packages
    pkg_checks = {
        "pandas":      "pandas",
        "numpy":       "numpy",
        "scipy":       "scipy",
        "statsmodels": "statsmodels",
        "econml":      "econml",
        "shap":        "shap",
        "sklearn":     "sklearn",
        "matplotlib":  "matplotlib",
        "openpyxl":    "openpyxl",
    }
    for display, pkg in pkg_checks.items():
        try:
            __import__(pkg)
            checks.append((f"Package: {display}", True, ""))
        except ImportError:
            checks.append((f"Package: {display}", False, "NOT INSTALLED"))

    # Print
    all_pass = True
    for desc, passed, note in checks:
        icon = "✓" if passed else "✗"
        line = f"  {icon} {desc}"
        if note:
            line += f"  [{note}]"
        print(line)
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  *** ALL CHECKS PASSED — study is ready to execute ***")
    else:
        print("  *** SOME CHECKS FAILED — see items marked ✗ above ***")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = run_preprocessing(verbose=True)
    readiness_check(df)
