"""
Synthetic data generator for the EPR governance experiment.

Variable naming follows the codebook exactly (survey/codebook/codebook.xlsx).
All construct items are on a 1–7 Likert scale unless stated otherwise.

Sections
--------
A  — Respondent information (POSITION, TENURE, FAMILIARITY)
B  — Firm characteristics (INDUSTRY, EMP_SIZE, REVENUE, EXPORT_PCT,
                            OWNERSHIP, SUPPLY_POS)
C  — Environmental management binary controls (ESG_REP, ISO14001,
                            ENV_INSP, ENV_PEN, EPR_REP)
D  — EPR Awareness: AWR1–AWR5  → AWR_MEAN
E  — Environmental Orientation: EO1–EO5  → EO_MEAN
F  — Compliance Capability: CC1–CC5  → CC_MEAN
G  — Resource Constraints: RC1–RC5  → RC_MEAN
H  — Baseline pressure exposures (pre-treatment):
       RP1–RP4  → RP_MEAN   (Regulatory)
       REP1–REP4 → REP_MEAN (Reputational)
       MP1–MP4  → MP_MEAN   (Market)
I  — Treatment assignment: TREAT (0=Control, 1=Regulatory,
                            2=Reputational, 3=Market)
J  — Manipulation check: MC1–MC3
K  — Budget allocation exercise: BAE_MKT, BAE_PROD, BAE_OPS,
                            BAE_TRAIN, BAE_EPR  (sum = 100; primary DV)
L  — Compliance Intentions: CI1–CI5  → CI_MEAN  (primary DV)
M  — Policy Preferences: PP1–PP4  → PP_MEAN

Usage
-----
    from data.synthetic.generate_epr_data import generate_epr_data
    df = generate_epr_data(n=800, seed=42)
    df.to_csv("data/synthetic/epr_survey_synthetic.csv", index=False)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical covariate list (used as X throughout the pipeline)
# Matches codebook Section B–H construct means / binary variables
# ---------------------------------------------------------------------------

COVARIATE_COLS = [
    # Section B — Firm characteristics
    "EMP_SIZE",         # Ordinal 1–6  (<10, 10–49, 50–99, 100–299, 300–999, 1000+)
    "REVENUE",          # Ordinal 1–5  (<1B, 1–10B, 10–100B, 100–500B, 500B+ KRW)
    "EXPORT_PCT",       # Continuous 0–100 (% of sales from exports)
    "OWNERSHIP",        # Categorical 1–6
    "SUPPLY_POS",       # Categorical 1–6
    # Section C — Environmental management
    "ESG_REP",          # Binary 0/1
    "ISO14001",         # Binary 0/1
    "ENV_INSP",         # Binary 0/1 (inspection past 3 yrs)
    "ENV_PEN",          # Binary 0/1 (penalty past 3 yrs)
    "EPR_REP",          # Binary 0/1 (EPR report previously submitted)
    # Section D–G — Construct means (1–7 scale)
    "AWR_MEAN",         # EPR Awareness
    "EO_MEAN",          # Environmental Orientation
    "CC_MEAN",          # Compliance Capability
    "RC_MEAN",          # Resource Constraints
    # Section H — Baseline pressure exposure means (1–7 scale)
    "RP_MEAN",          # Regulatory Pressure (baseline)
    "REP_MEAN",         # Reputational Pressure (baseline)
    "MP_MEAN",          # Market Pressure (baseline)
]

TREATMENT_LABELS = {
    0: "Control",
    1: "Regulatory",
    2: "Reputational",
    3: "Market",
}

# Item-level column groups (useful for reliability / CFA checks)
ITEM_GROUPS = {
    "AWR":  [f"AWR{i}" for i in range(1, 6)],
    "EO":   [f"EO{i}"  for i in range(1, 6)],
    "CC":   [f"CC{i}"  for i in range(1, 6)],
    "RC":   [f"RC{i}"  for i in range(1, 6)],
    "RP":   [f"RP{i}"  for i in range(1, 5)],
    "REP":  [f"REP{i}" for i in range(1, 5)],
    "MP":   [f"MP{i}"  for i in range(1, 5)],
    "CI":   [f"CI{i}"  for i in range(1, 6)],
    "PP":   [f"PP{i}"  for i in range(1, 5)],
    "MC":   [f"MC{i}"  for i in range(1, 4)],
    "BAE":  ["BAE_MKT", "BAE_PROD", "BAE_OPS", "BAE_TRAIN", "BAE_EPR"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _likert7(rng, mu, sd, n, n_items=1):
    """Draw n_items correlated 1–7 Likert responses around latent mu."""
    items = np.clip(np.round(rng.normal(mu, sd, (n, n_items))), 1, 7).astype(int)
    return items


def _true_tau(X: pd.DataFrame, arm: int) -> np.ndarray:
    """
    Latent CATE on the BAE_EPR scale (percentage points), theory-driven:
      T1 Regulatory   — ENV_INSP, CC_MEAN, RP_MEAN amplify
      T2 Reputational — EO_MEAN, ESG_REP, REP_MEAN amplify
      T3 Market       — EXPORT_PCT, SUPPLY_POS (downstream=6), MP_MEAN amplify
    """
    n = len(X)
    if arm == 1:
        tau = (3.0
               + 2.0 * X["ENV_INSP"]
               + 1.5 * (X["CC_MEAN"] - 4) / 3
               - 1.0 * (X["RC_MEAN"] - 4) / 3
               + 1.0 * (X["RP_MEAN"] - 4) / 3)
    elif arm == 2:
        tau = (2.5
               + 2.0 * (X["EO_MEAN"] - 4) / 3
               + 1.5 * X["ESG_REP"]
               + 1.0 * (X["REP_MEAN"] - 4) / 3)
    elif arm == 3:
        tau = (2.5
               + 0.05 * X["EXPORT_PCT"]          # each 1 ppt export share
               + 1.5 * (X["SUPPLY_POS"] >= 5).astype(float)   # distributor/retailer
               + 1.0 * (X["MP_MEAN"] - 4) / 3)
    else:
        tau = np.zeros(n)
    return tau.values if hasattr(tau, "values") else np.array(tau)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_epr_data(n: int = 800, seed: int = 42,
                      equal_arms: bool = True) -> pd.DataFrame:
    """
    Generate a synthetic EPR survey experiment dataset that mirrors
    the codebook structure exactly.

    Parameters
    ----------
    n          : Total number of firm respondents.
    seed       : Random seed.
    equal_arms : Assign equal n/4 per arm if True.

    Returns
    -------
    pd.DataFrame — one row per firm, columns matching the codebook.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Section A — Respondent information
    # ------------------------------------------------------------------
    POSITION   = rng.integers(1, 8, n)      # 1–7 categories
    TENURE     = np.clip(
        np.round(rng.gamma(3, 4, n)), 1, 35
    ).astype(int)
    FAMILIARITY = _likert7(rng, 4.5, 1.2, n)[:, 0]   # EPR familiarity 1–7

    # ------------------------------------------------------------------
    # Section B — Firm characteristics
    # ------------------------------------------------------------------
    INDUSTRY   = rng.integers(1, 8, n)      # 1=Electronics … 7=Other
    EMP_SIZE   = rng.choice([1,2,3,4,5,6], n,
                             p=[0.12, 0.22, 0.18, 0.25, 0.15, 0.08])
    REVENUE    = rng.choice([1,2,3,4,5], n,
                             p=[0.10, 0.28, 0.35, 0.18, 0.09])
    # Export share: right-skewed, many firms export little
    EXPORT_PCT = np.clip(
        rng.beta(1.2, 4.5, n) * 100, 0, 100
    ).round(1)
    OWNERSHIP  = rng.choice([1,2,3,4,5,6], n,
                             p=[0.35, 0.20, 0.15, 0.15, 0.10, 0.05])
    SUPPLY_POS = rng.choice([1,2,3,4,5,6], n,
                             p=[0.20, 0.25, 0.20, 0.15, 0.12, 0.08])

    # ------------------------------------------------------------------
    # Section C — Environmental management binaries
    # ------------------------------------------------------------------
    # Larger firms more likely to have ESG/ISO
    esg_p  = np.clip(0.12 + 0.08 * (EMP_SIZE - 3), 0.05, 0.85)
    iso_p  = np.clip(0.10 + 0.07 * (EMP_SIZE - 3), 0.04, 0.80)
    ESG_REP  = rng.binomial(1, esg_p)
    ISO14001 = rng.binomial(1, iso_p)
    ENV_INSP = rng.binomial(1, 0.35, n)
    ENV_PEN  = rng.binomial(1, 0.12, n)   # fewer receive penalties
    EPR_REP  = rng.binomial(1, 0.58, n)   # majority have submitted

    # ------------------------------------------------------------------
    # Section D–G — Multi-item constructs (5 items each, 1–7)
    # ------------------------------------------------------------------
    # Latent means correlated with firm size and ownership
    base_awareness  = 4.0 + 0.2 * (EMP_SIZE - 3) + 0.3 * EPR_REP
    base_eo         = 3.8 + 0.2 * (EMP_SIZE - 3) + 0.5 * ESG_REP
    base_cc         = 3.6 + 0.3 * (EMP_SIZE - 3) + 0.4 * ISO14001
    base_rc         = 4.0 - 0.2 * (EMP_SIZE - 3)   # larger firms → fewer constraints

    AWR_items  = np.clip(np.round(
        base_awareness[:, None] + rng.normal(0, 0.8, (n, 5))), 1, 7).astype(int)
    EO_items   = np.clip(np.round(
        base_eo[:, None]        + rng.normal(0, 0.9, (n, 5))), 1, 7).astype(int)
    CC_items   = np.clip(np.round(
        base_cc[:, None]        + rng.normal(0, 0.8, (n, 5))), 1, 7).astype(int)
    RC_items   = np.clip(np.round(
        base_rc[:, None]        + rng.normal(0, 0.9, (n, 5))), 1, 7).astype(int)

    AWR_MEAN = AWR_items.mean(axis=1)
    EO_MEAN  = EO_items.mean(axis=1)
    CC_MEAN  = CC_items.mean(axis=1)
    RC_MEAN  = RC_items.mean(axis=1)

    # ------------------------------------------------------------------
    # Section H — Baseline pressure exposures (4 items each, 1–7)
    # ------------------------------------------------------------------
    base_rp  = 3.5 + 0.3 * ENV_INSP
    base_rep = 3.2 + 0.4 * ESG_REP
    base_mp  = 2.8 + 0.04 * EXPORT_PCT

    RP_items  = np.clip(np.round(
        base_rp[:, None]  + rng.normal(0, 0.9, (n, 4))), 1, 7).astype(int)
    REP_items = np.clip(np.round(
        base_rep[:, None] + rng.normal(0, 0.9, (n, 4))), 1, 7).astype(int)
    MP_items  = np.clip(np.round(
        base_mp[:, None]  + rng.normal(0, 0.9, (n, 4))), 1, 7).astype(int)

    RP_MEAN  = RP_items.mean(axis=1)
    REP_MEAN = REP_items.mean(axis=1)
    MP_MEAN  = MP_items.mean(axis=1)

    # ------------------------------------------------------------------
    # Section I — Treatment assignment
    # ------------------------------------------------------------------
    if equal_arms:
        arms = np.repeat([0, 1, 2, 3], n // 4)
        remainder = n - len(arms)
        arms = np.concatenate([arms, rng.integers(0, 4, remainder)])
        rng.shuffle(arms)
    else:
        arms = rng.integers(0, 4, n)

    TREAT = arms

    # ------------------------------------------------------------------
    # Section J — Manipulation check (MC1–MC3, 1–7)
    # ------------------------------------------------------------------
    # MC1 probes regulatory salience, MC2 reputational, MC3 market
    mc_base = np.where(TREAT == 1, 5.2,
              np.where(TREAT == 2, 4.0,
              np.where(TREAT == 3, 4.0, 3.5)))
    MC_items = np.clip(np.round(
        mc_base[:, None] + rng.normal(0, 0.8, (n, 3))), 1, 7).astype(int)
    # Arm-specific salience boosts
    MC_items[TREAT == 1, 0] = np.clip(MC_items[TREAT == 1, 0] + 1, 1, 7)
    MC_items[TREAT == 2, 1] = np.clip(MC_items[TREAT == 2, 1] + 1, 1, 7)
    MC_items[TREAT == 3, 2] = np.clip(MC_items[TREAT == 3, 2] + 1, 1, 7)

    # ------------------------------------------------------------------
    # Section K — Budget Allocation Exercise (sum-constrained to 100)
    # ------------------------------------------------------------------
    # Covariates DataFrame for CATE computation
    X_df = pd.DataFrame({
        "EMP_SIZE": EMP_SIZE, "REVENUE": REVENUE,
        "EXPORT_PCT": EXPORT_PCT, "OWNERSHIP": OWNERSHIP,
        "SUPPLY_POS": SUPPLY_POS, "ESG_REP": ESG_REP,
        "ISO14001": ISO14001, "ENV_INSP": ENV_INSP,
        "ENV_PEN": ENV_PEN, "EPR_REP": EPR_REP,
        "AWR_MEAN": AWR_MEAN, "EO_MEAN": EO_MEAN,
        "CC_MEAN": CC_MEAN, "RC_MEAN": RC_MEAN,
        "RP_MEAN": RP_MEAN, "REP_MEAN": REP_MEAN,
        "MP_MEAN": MP_MEAN,
    })

    # Baseline EPR budget share (%)
    baseline_epr = (
        6.0
        + 1.5 * (EMP_SIZE - 3)
        + 1.2 * (CC_MEAN - 4)
        + 0.8 * (AWR_MEAN - 4)
        + 1.5 * ESG_REP
    )

    # True CATE (% points) per arm
    tau = np.zeros(n)
    for arm in [1, 2, 3]:
        mask = TREAT == arm
        tau[mask] = _true_tau(X_df[mask], arm)

    noise_epr = rng.normal(0, 2.5, n)
    BAE_EPR_raw = np.clip(baseline_epr + tau + noise_epr, 0, 60)

    # Distribute remaining budget among other 4 categories via Dirichlet
    other_total = 100 - BAE_EPR_raw
    other_props = rng.dirichlet(alpha=[3, 4, 2, 1], size=n)   # MKT, PROD, OPS, TRAIN
    BAE_MKT   = (other_props[:, 0] * other_total).round(1)
    BAE_PROD  = (other_props[:, 1] * other_total).round(1)
    BAE_OPS   = (other_props[:, 2] * other_total).round(1)
    BAE_TRAIN = (100 - BAE_EPR_raw.round(1) - BAE_MKT - BAE_PROD - BAE_OPS).round(1)
    BAE_EPR   = BAE_EPR_raw.round(1)

    # ------------------------------------------------------------------
    # Section L — Compliance Intentions (CI1–CI5, 1–7)
    # ------------------------------------------------------------------
    latent_ci = (
        3.5
        + 0.35 * (EO_MEAN - 4)
        + 0.30 * (AWR_MEAN - 4)
        + 0.20 * (CC_MEAN - 4)
        - 0.20 * (RC_MEAN - 4)
        + tau * 0.20      # treatment bump scaled to 7-pt range
    )
    CI_items = np.clip(np.round(
        latent_ci[:, None] + rng.normal(0, 0.8, (n, 5))), 1, 7).astype(int)
    CI_MEAN  = CI_items.mean(axis=1)

    # ------------------------------------------------------------------
    # Section M — Policy Preferences (PP1–PP4, 1–7)
    # ------------------------------------------------------------------
    latent_pp = 3.8 + 0.25 * (EO_MEAN - 4) + tau * 0.10
    PP_items = np.clip(np.round(
        latent_pp[:, None] + rng.normal(0, 0.9, (n, 4))), 1, 7).astype(int)
    PP_MEAN  = PP_items.mean(axis=1)

    # ------------------------------------------------------------------
    # Assemble full DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        "firm_id": np.arange(1, n + 1),
        # Section A
        "POSITION":    POSITION,
        "TENURE":      TENURE,
        "FAMILIARITY": FAMILIARITY,
        # Section B
        "INDUSTRY":  INDUSTRY,
        "EMP_SIZE":  EMP_SIZE,
        "REVENUE":   REVENUE,
        "EXPORT_PCT": EXPORT_PCT,
        "OWNERSHIP": OWNERSHIP,
        "SUPPLY_POS": SUPPLY_POS,
        # Section C
        "ESG_REP":  ESG_REP,
        "ISO14001":  ISO14001,
        "ENV_INSP":  ENV_INSP,
        "ENV_PEN":   ENV_PEN,
        "EPR_REP":   EPR_REP,
    })

    # Append item-level columns
    for i, col in enumerate(ITEM_GROUPS["AWR"]):
        df[col] = AWR_items[:, i]
    df["AWR_MEAN"] = AWR_MEAN.round(3)

    for i, col in enumerate(ITEM_GROUPS["EO"]):
        df[col] = EO_items[:, i]
    df["EO_MEAN"] = EO_MEAN.round(3)

    for i, col in enumerate(ITEM_GROUPS["CC"]):
        df[col] = CC_items[:, i]
    df["CC_MEAN"] = CC_MEAN.round(3)

    for i, col in enumerate(ITEM_GROUPS["RC"]):
        df[col] = RC_items[:, i]
    df["RC_MEAN"] = RC_MEAN.round(3)

    for i, col in enumerate(ITEM_GROUPS["RP"]):
        df[col] = RP_items[:, i]
    df["RP_MEAN"] = RP_MEAN.round(3)

    for i, col in enumerate(ITEM_GROUPS["REP"]):
        df[col] = REP_items[:, i]
    df["REP_MEAN"] = REP_MEAN.round(3)

    for i, col in enumerate(ITEM_GROUPS["MP"]):
        df[col] = MP_items[:, i]
    df["MP_MEAN"] = MP_MEAN.round(3)

    # Treatment
    df["TREAT"] = TREAT
    df["TREAT_LABEL"] = df["TREAT"].map(TREATMENT_LABELS)

    # Manipulation check
    for i, col in enumerate(ITEM_GROUPS["MC"]):
        df[col] = MC_items[:, i]

    # Budget allocation
    df["BAE_MKT"]   = BAE_MKT
    df["BAE_PROD"]  = BAE_PROD
    df["BAE_OPS"]   = BAE_OPS
    df["BAE_TRAIN"] = BAE_TRAIN
    df["BAE_EPR"]   = BAE_EPR     # primary DV

    # Compliance intentions
    for i, col in enumerate(ITEM_GROUPS["CI"]):
        df[col] = CI_items[:, i]
    df["CI_MEAN"] = CI_MEAN.round(3)

    # Policy preferences
    for i, col in enumerate(ITEM_GROUPS["PP"]):
        df[col] = PP_items[:, i]
    df["PP_MEAN"] = PP_MEAN.round(3)

    return df


if __name__ == "__main__":
    import os
    df = generate_epr_data(n=800, seed=42)
    out = os.path.join(os.path.dirname(__file__), "epr_survey_synthetic.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows → {out}")
    print("\nMean outcomes by arm:")
    print(df.groupby("TREAT_LABEL")[["BAE_EPR", "CI_MEAN", "PP_MEAN"]].mean().round(3))
    print(f"\nTotal columns: {len(df.columns)}")
