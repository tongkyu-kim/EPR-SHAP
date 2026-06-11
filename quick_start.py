"""
Quick-Start Guide: CausalForestDML + SHAP Implementation

This script demonstrates how to use the complete workflow files.
"""

# ============================================================================
# INSTALLATION
# ============================================================================
# Run these commands in terminal:
#
# pip install econml scikit-learn shap pandas numpy matplotlib seaborn
#
# For conda users:
# conda install -c conda-forge econml shap scikit-learn pandas numpy matplotlib seaborn
#

# ============================================================================
# QUICK START: 3 LINES OF CODE
# ============================================================================

from causal_forest_shap_workflow import run_complete_workflow

# Run the complete analysis pipeline
results = run_complete_workflow(n_samples=500, n_features=10, random_state=42)

# Access key results
model = results['model']
cate_estimates = results['cate_estimates']  # Heterogeneous treatment effects
shap_values = results['shap_values']        # Feature importance for effects
figures = results['figures']                # All visualizations


# ============================================================================
# DETAILED WORKFLOW: STEP-BY-STEP
# ============================================================================

import numpy as np
import pandas as pd
from causal_forest_shap_workflow import (
    generate_treatment_effect_data,
    fit_causal_forest_dml,
    estimate_conditional_treatment_effects,
    extract_shap_values_from_causal_forest,
    plot_shap_summary_beeswarm,
    plot_shap_dependence_plots,
    plot_shap_force_plots
)

# Step 1: Generate or load your data
X, T, Y, true_effects, feature_names = generate_treatment_effect_data(
    n_samples=1000,
    n_features=15,
    effect_type='heterogeneous'
)

# Step 2: Fit causal forest model
model, X_train, X_test, T_train, T_test, Y_train, Y_test = fit_causal_forest_dml(
    X, T, Y, test_size=0.2, random_state=42
)

# Step 3: Estimate heterogeneous treatment effects
cate_estimates, cate_stderr, cate_ci = estimate_conditional_treatment_effects(
    model, X, X_test
)
print(f"Average treatment effect: {cate_estimates.mean():.4f}")

# Step 4: Extract SHAP values for interpretation
shap_values, base_value, explainer = extract_shap_values_from_causal_forest(
    model, X_test, n_samples=100
)

# Step 5: Create visualizations
fig_summary = plot_shap_summary_beeswarm(shap_values, X_test, feature_names)
fig_dependence = plot_shap_dependence_plots(shap_values, X_test, feature_names)
fig_force = plot_shap_force_plots(shap_values, X_test, feature_names, n_samples=3)


# ============================================================================
# ADVANCED ANALYSIS: USE EXTENDED EXAMPLES
# ============================================================================

from advanced_examples import (
    analyze_heterogeneous_effects_by_firm_type,
    plot_treatment_effect_curves,
    sensitivity_analysis_unmeasured_confounder,
    compare_predicted_vs_actual_effects,
    generate_targeting_recommendations,
    run_all_advanced_examples
)

# Analyze treatment effects by firm subgroups
effects_by_size, fig_by_size = analyze_heterogeneous_effects_by_firm_type(
    model, X_test, cate_estimates, feature_names, feature_idx_firm_size=0
)

# Plot how effects vary with firm size
fig_curves = plot_treatment_effect_curves(
    X_test, cate_estimates, feature_names, feature_idx=0, n_bins=20
)

# Sensitivity analysis: robustness to unmeasured confounding
sensitivity, fig_sens = sensitivity_analysis_unmeasured_confounder(
    model, X_test, feature_names
)

# Compare predictions against true effects (if available)
metrics, fig_pred = compare_predicted_vs_actual_effects(
    cate_estimates, true_effects
)

# Generate policy targeting recommendations
targeting_df, fig_targeting = generate_targeting_recommendations(
    X_test, cate_estimates, feature_names, quantile_threshold=0.75
)

# Or run all examples at once
all_results = run_all_advanced_examples(
    model=model,
    X_test=X_test,
    cate_estimates=cate_estimates,
    cate_stderr=cate_stderr,
    true_effects=true_effects,
    X_train=X_train,
    T_train=T_train,
    Y_train=Y_train,
    feature_names=feature_names,
    shap_values=shap_values
)


# ============================================================================
# USING YOUR OWN DATA
# ============================================================================

# Load your firm-level data
# Assuming CSV format with columns:
# - Covariates: [feature_1, feature_2, ..., feature_n]
# - Treatment: binary column (0=control, 1=treatment)
# - Outcome: numerical column (firm performance metric)

import pandas as pd

df = pd.read_csv('your_firm_data.csv')

# Prepare data matrices
feature_names = ['firm_size', 'rd_intensity', 'market_share', 'age', 'exports']
X = df[feature_names].values
T = df['policy_participation'].values
Y = df['productivity_growth'].values

# Now use the workflow as above
model, X_train, X_test, T_train, T_test, Y_train, Y_test = fit_causal_forest_dml(
    X, T, Y
)
cate_estimates, _, _ = estimate_conditional_treatment_effects(model, X, X_test)


# ============================================================================
# SAVING AND VISUALIZING RESULTS
# ============================================================================

import matplotlib.pyplot as plt

# Save all figures
figures_to_save = [
    ('beeswarm_plot.png', results['figures']['beeswarm']),
    ('dependence_plots.png', results['figures']['dependence']),
    ('importance_plot.png', results['figures']['importance'])
]

for filename, fig in figures_to_save:
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")

# Or save as PDF for reports
results['figures']['beeswarm'].savefig('analysis_summary.pdf', bbox_inches='tight')

# Export results to CSV
results_df = pd.DataFrame({
    'predicted_effect': cate_estimates,
    'effect_stderr': cate_stderr,
    'firm_size': X_test[:, 0],
    'rd_intensity': X_test[:, 1],
    'market_share': X_test[:, 2]
})
results_df.to_csv('treatment_effects_predictions.csv', index=False)


# ============================================================================
# KEY INTERPRETATION GUIDELINES
# ============================================================================

"""
HETEROGENEOUS TREATMENT EFFECTS (CATE):
- Positive CATE: Treatment increases outcome (benefit for that firm)
- Negative CATE: Treatment decreases outcome (harm for that firm)
- Std Error: Measure of uncertainty; wider CI = less certain

SHAP FEATURE IMPORTANCE:
- Larger |SHAP value| = Feature more important for predicting effects
- Beeswarm plot shows full distribution of effects

SHAP DEPENDENCE PLOT:
- Shows relationship between feature value and treatment effect
- Red dots: feature increases treatment effect
- Blue dots: feature decreases treatment effect
- Automatically colored by interacting feature

SENSITIVITY ANALYSIS:
- t-statistic > 2: Results robust to moderate unmeasured confounding
- t-statistic < 1: Results may be sensitive to confounding

POLICY TARGETING:
- High-effect firms are best candidates for limited program budget
- Characteristic analysis shows which firm types benefit most
"""


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Q: "ModuleNotFoundError: No module named 'econml'"
A: Install with: pip install econml

Q: "Why are my SHAP values all zeros?"
A: Ensure the causal forest model fitted successfully. Check:
   - Sufficient variation in treatment variable (T)
   - No collinear features
   - Adequate sample size (n > 100 recommended)

Q: "How do I use this with panel data?"
A: For panel data, use time-period fixed effects in outcome model:
   - Add time dummies to X
   - Use econml.dml.DMLOrthoForest for more flexibility

Q: "Can I use continuous treatment instead of binary?"
A: Yes, just pass continuous T to fit(). Model will estimate
   dose-response effects. Interpretation: effect per unit increase in T.

Q: "How do I validate the model?"
A: Use cross-validation; split data for proper validation; compare
   predicted vs actual effects if ground truth available.
"""


# ============================================================================
# RESOURCES AND CITATIONS
# ============================================================================

"""
KEY PAPERS:

1. Athey & Wager (2019) - Generalized Random Forests
   https://arxiv.org/abs/1610.01271
   Foundation for causal forests and heterogeneous effects

2. Chernozhukov et al. (2018) - Double Machine Learning
   https://arxiv.org/abs/1707.09519
   Framework for combining ML with causal inference

3. Lundberg et al. (2020) - From Local Explanations to Global
   Understanding with SHAP
   https://arxiv.org/abs/1905.04610
   SHAP values and TreeSHAP for tree models

4. Kennedy (2023) - Towards optimal doubly robust estimation
   of heterogeneous causal effects
   https://arxiv.org/abs/2004.14497

DOCUMENTATION:
- EconML: https://www.pywhy.org/EconML/
- SHAP: https://shap.readthedocs.io/
- Scikit-Learn: https://scikit-learn.org/
"""

if __name__ == "__main__":
    print(__doc__)
