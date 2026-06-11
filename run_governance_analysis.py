"""
Full governance analysis pipeline.

Stages
------
1  ATE estimation     (ANCOVA + DiM)
2  CATE estimation    (pairwise CausalForestDML)
3  Treatment ranking  (best_treatment per firm)
4  Subgroup analysis  (heterogeneity by firm characteristics)
5  Archetypes         (k-means clustering + profiles)
6  Policy simulation  (scenarios A-D)
7  SHAP               (explain CATE heterogeneity per arm)
                      -- skipped by default; pass  --shap  to enable --

Usage
-----
    python run_governance_analysis.py           # stages 1-6
    python run_governance_analysis.py --shap    # stages 1-7 (adds ~15 min)
    python run_governance_analysis.py --shap --n-clusters 5
"""

import sys
import os
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project imports ──────────────────────────────────────────────────────────
from src.preprocessing.data_cleaning import load_synthetic
from src.ate.ate_estimation         import estimate_all_ates, ate_summary_table
from src.cate.causal_forest_epr     import run_cate_pipeline, plot_cate_distributions

from src.governance.treatment_ranking import (
    compute_treatment_ranking,
    ranking_summary,
    plot_ranking_frequencies,
    plot_cate_pairwise_scatter,
    plot_gain_distribution,
)
from src.governance.subgroup_analysis import (
    subgroup_cate_table,
    subgroup_ttest_table,
    plot_subgroup_heatmap,
    plot_subgroup_bar_panel,
)
from src.governance.archetypes import (
    fit_archetypes,
    profile_archetypes,
    archetype_treatment_effects,
    name_archetypes,
    elbow_scores,
    plot_archetype_profiles,
    plot_archetype_cates,
    plot_archetype_pca,
    plot_elbow,
)
from src.governance.policy_simulation import (
    simulate_scenarios,
    plot_scenario_comparison,
    plot_optimal_assignment_map,
    plot_assignment_breakdown,
)

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shap",        action="store_true",
                   help="Run SHAP stage (slow, ~15 min)")
    p.add_argument("--n-clusters",  type=int, default=4,
                   help="Number of archetype clusters (default: 4)")
    p.add_argument("--n-estimators",type=int, default=300,
                   help="Causal forest trees (default: 300)")
    p.add_argument("--out",         type=str,
                   default=os.path.join(ROOT, "outputs", "governance"),
                   help="Output directory")
    return p.parse_args()

# ── helpers ───────────────────────────────────────────────────────────────────

def savefig(fig, path):
    if fig is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def section(title):
    print(f"\n{'='*72}\n{title}\n{'='*72}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    OUT        = args.out
    TABLES_DIR = os.path.join(OUT, "tables")
    FIGS_DIR   = os.path.join(OUT, "figures")
    SHAP_DIR   = os.path.join(OUT, "shap")
    for d in [TABLES_DIR, FIGS_DIR, SHAP_DIR]:
        os.makedirs(d, exist_ok=True)

    DATA_PATH = os.path.join(ROOT, "data", "processed", "survey_data_clean.csv")

    # ── Load data ─────────────────────────────────────────────────────────────
    section("LOADING DATA")
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"  Loaded survey_data_clean.csv  ({len(df)} rows)")
    else:
        print("  survey_data_clean.csv not found -- using synthetic data (n=600)")
        df, _ = load_synthetic(n=600, seed=42)

    print(f"  Arm counts:\n{df['TREAT'].value_counts().sort_index().to_string()}")

    # ── Stage 1: ATE ─────────────────────────────────────────────────────────
    section("STAGE 1: ATE ESTIMATION")
    ate_results = estimate_all_ates(df)
    ate_tbl     = ate_summary_table(ate_results)
    ate_tbl.to_csv(os.path.join(TABLES_DIR, "ate_results.csv"), index=False)
    print(f"\n  Saved: tables/ate_results.csv")

    # ── Stage 2: CATE ─────────────────────────────────────────────────────────
    section("STAGE 2: CATE ESTIMATION (Pairwise CausalForestDML)")
    models, df_cate, hypotheses = run_cate_pipeline(
        df, outcome="BAE_EPR", mode="pairwise",
        n_estimators=args.n_estimators, random_state=42,
    )
    hypotheses.to_csv(os.path.join(TABLES_DIR, "cate_hypothesis_tests.csv"), index=False)

    fig = plot_cate_distributions(df_cate, figsize=(14, 4))
    savefig(fig, os.path.join(FIGS_DIR, "cate_distributions.png"))
    print("  Saved: figures/cate_distributions.png")

    cate_cols = [c for c in df_cate.columns
                 if c.startswith("CATE_") and not c.endswith("_SE")]
    print("\n  CATE summary (mean per arm):")
    for col in cate_cols:
        arm = col.replace("CATE_", "").title()
        print(f"    {arm:14s}  mean={df_cate[col].mean():+.3f}  "
              f"sd={df_cate[col].std():.3f}")

    # ── Stage 3: Treatment ranking ────────────────────────────────────────────
    section("STAGE 3: TREATMENT RANKING")
    df_ranked = compute_treatment_ranking(df_cate)
    rank_sum  = ranking_summary(df_ranked)

    print("\n  Ranking summary:")
    print(rank_sum.to_string())

    rank_sum.to_csv(os.path.join(TABLES_DIR, "treatment_ranking_summary.csv"))
    df_ranked[["best_treatment", "best_cate", "gain_over_mean",
               "gain_over_worst",
               "rank_enforcement", "rank_reputation", "rank_market"]
              ].to_csv(os.path.join(TABLES_DIR, "treatment_ranking_full.csv"),
                       index=False)

    savefig(plot_ranking_frequencies(df_ranked),
            os.path.join(FIGS_DIR, "ranking_frequencies.png"))
    savefig(plot_cate_pairwise_scatter(df_ranked),
            os.path.join(FIGS_DIR, "cate_pairwise_scatter.png"))
    savefig(plot_gain_distribution(df_ranked),
            os.path.join(FIGS_DIR, "gain_distribution.png"))
    print("  Saved: 3 treatment-ranking figures")

    # ── Stage 4: Subgroup analysis ────────────────────────────────────────────
    section("STAGE 4: SUBGROUP HETEROGENEITY ANALYSIS")
    sg_table  = subgroup_cate_table(df_ranked)
    ttest_tbl = subgroup_ttest_table(df_ranked)

    sg_table.to_csv(os.path.join(TABLES_DIR, "subgroup_cate_means.csv"),  index=False)
    ttest_tbl.to_csv(os.path.join(TABLES_DIR, "subgroup_ttests.csv"),     index=False)

    print("\n  Subgroup t-tests (binary comparisons):")
    print(ttest_tbl[["Subgroup", "Arm", "Difference", "p_value", "Sig"]]
          .to_string(index=False))

    savefig(plot_subgroup_heatmap(sg_table),
            os.path.join(FIGS_DIR, "subgroup_heatmap.png"))
    savefig(plot_subgroup_bar_panel(sg_table),
            os.path.join(FIGS_DIR, "subgroup_bar_panel.png"))
    print("\n  Saved: 2 subgroup figures")

    # ── Stage 5: Archetypes ───────────────────────────────────────────────────
    section(f"STAGE 5: GOVERNANCE ARCHETYPES  (k={args.n_clusters})")

    # Elbow plot to validate cluster count choice
    elbow_df = elbow_scores(df_ranked, k_range=range(2, 9))
    savefig(plot_elbow(elbow_df), os.path.join(FIGS_DIR, "archetype_elbow.png"))

    df_arch, km, scaler, feat_used = fit_archetypes(
        df_ranked, n_clusters=args.n_clusters
    )
    profile = profile_archetypes(df_arch, features=feat_used)
    te      = archetype_treatment_effects(df_arch)
    names   = name_archetypes(profile, te)

    print("\n  Archetype profiles:")
    print(profile.to_string())
    print("\n  Archetype treatment effects:")
    print(te.to_string())
    print("\n  Auto-generated names:")
    for arch, name in names.items():
        print(f"    {arch}: {name}")

    profile.to_csv(os.path.join(TABLES_DIR, "archetype_profiles.csv"))
    te.to_csv(os.path.join(TABLES_DIR, "archetype_treatment_effects.csv"))
    pd.DataFrame.from_dict(names, orient="index",
                           columns=["Description"]).to_csv(
        os.path.join(TABLES_DIR, "archetype_names.csv")
    )

    savefig(plot_archetype_profiles(profile),
            os.path.join(FIGS_DIR, "archetype_profiles.png"))
    savefig(plot_archetype_cates(te),
            os.path.join(FIGS_DIR, "archetype_cates.png"))
    savefig(plot_archetype_pca(df_arch, features=feat_used),
            os.path.join(FIGS_DIR, "archetype_pca.png"))
    print("\n  Saved: 3 archetype figures (+ elbow)")

    # ── Stage 6: Policy simulation ────────────────────────────────────────────
    section("STAGE 6: POLICY SIMULATION (Scenarios A-D)")
    scenario_df, assign_df = simulate_scenarios(df_ranked)

    print("\n  Scenario comparison:")
    print(scenario_df[["Scenario", "Mean CATE (pp)",
                        "Expected BAE_EPR",
                        "Gain vs. best universal (pp)"]].to_string(index=False))
    print("\n  Scenario D assignment breakdown:")
    print(assign_df.to_string(index=False))

    scenario_df.to_csv(os.path.join(TABLES_DIR, "policy_scenarios.csv"),    index=False)
    assign_df.to_csv(os.path.join(TABLES_DIR, "policy_assignment_d.csv"),   index=False)

    savefig(plot_scenario_comparison(scenario_df),
            os.path.join(FIGS_DIR, "policy_scenario_comparison.png"))
    savefig(plot_assignment_breakdown(assign_df),
            os.path.join(FIGS_DIR, "policy_assignment_donut.png"))

    # Assignment maps for two pairs of moderators
    for xc, yc in [("EXPORT_PCT", "EO_MEAN"), ("EMP_SIZE", "AWR_MEAN")]:
        savefig(plot_optimal_assignment_map(df_ranked, x_col=xc, y_col=yc),
                os.path.join(FIGS_DIR,
                             f"policy_assignment_map_{xc}_v_{yc}.png"))
    print("  Saved: 4 policy simulation figures")

    # ── Stage 7: SHAP (optional) ──────────────────────────────────────────────
    if args.shap:
        section("STAGE 7: SHAP ANALYSIS")
        from src.shap_analysis.shap_epr import run_shap_pipeline
        shap_results = run_shap_pipeline(
            models, df_ranked, save_dir=SHAP_DIR,
            bg_size=100, max_evals=500,
        )
        drivers = shap_results["top_drivers"]
        print("\n  Top 5 CATE drivers per arm:")
        for arm in drivers["Arm"].unique():
            sub = drivers[drivers["Arm"] == arm]
            print(f"\n  {arm}:")
            for _, row in sub.iterrows():
                print(f"    {row['Rank']}. {row['Feature']:14s}  "
                      f"mean|SHAP|={row['Mean |SHAP|']:.5f}  "
                      f"({row['Direction']})  {row['Interpretation']}")
    else:
        print("\n  SHAP stage skipped. Re-run with --shap to include it.")

    # ── Final summary ─────────────────────────────────────────────────────────
    section("ANALYSIS COMPLETE")
    print(f"\n  All outputs saved to: {OUT}")
    print(f"\n  Tables ({TABLES_DIR}):")
    for f in sorted(os.listdir(TABLES_DIR)):
        print(f"    {f}")
    print(f"\n  Figures ({FIGS_DIR}):")
    for f in sorted(os.listdir(FIGS_DIR)):
        print(f"    {f}")
    if args.shap and os.path.isdir(SHAP_DIR):
        print(f"\n  SHAP ({SHAP_DIR}):")
        for f in sorted(os.listdir(SHAP_DIR)):
            print(f"    {f}")


if __name__ == "__main__":
    main()
