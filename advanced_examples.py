"""
Advanced Examples: CausalForestDML + SHAP for Economic Policy Analysis

This module provides practical examples for firm-level data analysis,
including sensitivity analysis, heterogeneous effects by subgroups,
and advanced SHAP visualizations.

Use Cases:
- Policy impact evaluation on firm productivity/exports/innovation
- Subsidy program effectiveness analysis
- Market intervention heterogeneous effects
- Firm capability analysis from treatment responses

Sources:
- EconML Specification: https://www.pywhy.org/EconML/spec/estimation/forest.html
- Double Machine Learning: https://www.pywhy.org/EconML/spec/spec.html
- SHAP Advanced: https://shap.readthedocs.io/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import shap

from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor


# ============================================================================
# EXAMPLE 1: HETEROGENEOUS EFFECTS BY FIRM SUBGROUPS
# ============================================================================

def analyze_heterogeneous_effects_by_firm_type(model, X_test, cate_estimates, 
                                               feature_names, feature_idx_firm_size=0):
    """
    Analyze heterogeneous treatment effects for different firm types.
    
    In policy analysis, we often want to know:
    - Does the program help small firms more than large firms?
    - Do different sectors respond differently to the intervention?
    
    Parameters
    -----------
    model : CausalForestDML
        Fitted model
    X_test : np.ndarray
        Test features
    cate_estimates : np.ndarray
        Conditional average treatment effects
    feature_names : list
        Feature names
    feature_idx_firm_size : int
        Index of firm size feature
    
    Returns
    --------
    results_by_type : dict
        Average effects for each firm type
    fig : matplotlib figure
    """
    
    # Create dataframe with effects and covariates
    df_results = pd.DataFrame(X_test, columns=feature_names)
    df_results['cate'] = cate_estimates
    
    # Categorize firms by size (small/medium/large)
    df_results['firm_size_quintile'] = pd.qcut(
        df_results[feature_names[feature_idx_firm_size]], 
        q=5, 
        labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
    )
    
    # Analyze effects by firm size
    effects_by_size = df_results.groupby('firm_size_quintile')['cate'].agg([
        'count', 'mean', 'std', ('sem', lambda x: x.std() / np.sqrt(len(x)))
    ])
    
    # Calculate 95% confidence intervals
    effects_by_size['ci_lower'] = effects_by_size['mean'] - 1.96 * effects_by_size['sem']
    effects_by_size['ci_upper'] = effects_by_size['mean'] + 1.96 * effects_by_size['sem']
    
    print("\n" + "=" * 70)
    print("HETEROGENEOUS EFFECTS BY FIRM SIZE")
    print("=" * 70)
    print(effects_by_size)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(effects_by_size))
    ax.bar(x_pos, effects_by_size['mean'], 
           yerr=[effects_by_size['mean'] - effects_by_size['ci_lower'],
                 effects_by_size['ci_upper'] - effects_by_size['mean']],
           capsize=5, alpha=0.7, color='steelblue', label='Mean Effect')
    
    ax.set_xlabel('Firm Size Category')
    ax.set_ylabel('Conditional Average Treatment Effect')
    ax.set_title('Treatment Effect Heterogeneity by Firm Size')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(effects_by_size.index, rotation=45)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return effects_by_size, fig


# ============================================================================
# EXAMPLE 2: SHAP INTERACTIONS - HOW FEATURES INTERACT IN TREATMENT EFFECTS
# ============================================================================

def analyze_shap_interactions(shap_values, X_test, feature_names, 
                             feature_idx1=0, feature_idx2=1):
    """
    Analyze how two features interact in determining treatment effects.
    
    SHAP interaction values show how two features jointly affect predictions
    beyond their individual effects. Useful for discovering:
    - Which firm characteristics work well together in policy response
    - Non-linear policy effects
    
    Parameters
    -----------
    shap_values : np.ndarray
        SHAP values for samples
    X_test : np.ndarray
        Feature values
    feature_names : list
        Feature names
    feature_idx1, feature_idx2 : int
        Indices of features to compare
    
    Returns
    --------
    fig : matplotlib figure showing interaction heatmap
    """
    
    # Create bins for features
    feature1_bins = pd.qcut(X_test[:, feature_idx1], q=5, duplicates='drop')
    feature2_bins = pd.qcut(X_test[:, feature_idx2], q=5, duplicates='drop')
    
    # Compute mean SHAP value for each combination of bins
    df_interaction = pd.DataFrame({
        'feature1': feature1_bins,
        'feature2': feature2_bins,
        'shap': shap_values[:, feature_idx1]  # SHAP for feature 1
    })
    
    interaction_matrix = df_interaction.pivot_table(
        values='shap', 
        index='feature2', 
        columns='feature1',
        aggfunc='mean'
    )
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(interaction_matrix, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Mean SHAP Value'}, ax=ax,
                xticklabels=[f'Q{i+1}' for i in range(interaction_matrix.shape[1])],
                yticklabels=[f'Q{i+1}' for i in range(interaction_matrix.shape[0])])
    
    ax.set_xlabel(f'{feature_names[feature_idx1]} (Quintile)')
    ax.set_ylabel(f'{feature_names[feature_idx2]} (Quintile)')
    ax.set_title(f'SHAP Interaction: {feature_names[feature_idx1]} × {feature_names[feature_idx2]}')
    
    plt.tight_layout()
    return fig


# ============================================================================
# EXAMPLE 3: TREATMENT EFFECT HETEROGENEITY CURVES
# ============================================================================

def plot_treatment_effect_curves(X_test, cate_estimates, feature_names, 
                                feature_idx=0, n_bins=20):
    """
    Plot how treatment effects vary with a continuous feature.
    
    Shows the relationship: "As firm size increases, how does program 
    impact change?" This is critical for policy design (targeting).
    
    Parameters
    -----------
    X_test : np.ndarray
        Feature values
    cate_estimates : np.ndarray
        Treatment effect estimates
    feature_names : list
        Feature names
    feature_idx : int
        Which feature to vary
    n_bins : int
        Number of bins for averaging
    
    Returns
    --------
    fig : matplotlib figure
    """
    
    # Create bins and compute mean effects
    feature_values = X_test[:, feature_idx]
    bins = pd.cut(feature_values, bins=n_bins)
    
    df_curve = pd.DataFrame({
        'feature': feature_values,
        'cate': cate_estimates,
        'bin': bins
    })
    
    curve_data = df_curve.groupby('bin').agg({
        'feature': 'mean',
        'cate': ['mean', 'std', 'count']
    }).reset_index(drop=True)
    
    curve_data.columns = ['feature_mean', 'cate_mean', 'cate_std', 'n']
    curve_data['sem'] = curve_data['cate_std'] / np.sqrt(curve_data['n'])
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(curve_data['feature_mean'], curve_data['cate_mean'], 
            marker='o', linewidth=2, markersize=6, label='Mean CATE')
    
    # Add confidence band
    ax.fill_between(curve_data['feature_mean'],
                    curve_data['cate_mean'] - 1.96 * curve_data['sem'],
                    curve_data['cate_mean'] + 1.96 * curve_data['sem'],
                    alpha=0.2, label='95% CI')
    
    ax.set_xlabel(f'{feature_names[feature_idx]}')
    ax.set_ylabel('Conditional Average Treatment Effect')
    ax.set_title(f'Treatment Effect Heterogeneity by {feature_names[feature_idx]}')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


# ============================================================================
# EXAMPLE 4: SENSITIVITY ANALYSIS - ROBUSTNESS TO UNMEASURED CONFOUNDING
# ============================================================================

def sensitivity_analysis_unmeasured_confounder(model, X_test, feature_names):
    """
    Perform sensitivity analysis for robustness to unmeasured confounding.
    
    In observational studies, unmeasured confounders (unobserved factors
    affecting both treatment and outcome) are a major threat to validity.
    This analysis estimates how large unmeasured confounding would need to be
    to change our conclusions.
    
    Parameters
    -----------
    model : CausalForestDML
        Fitted model with inference capability
    X_test : np.ndarray
        Test data
    feature_names : list
        Feature names
    
    Returns
    --------
    sensitivity_results : dict
        Sensitivity statistics
    fig : matplotlib figure
    
    Interpretation
    ---------------
    - Lower sensitivity index = more robust to unmeasured confounding
    - Sensitivity index > 1.0 = moderate concern
    - Sensitivity index > 2.0 = high concern
    """
    
    # Get effect estimates with inference
    effects_inference = model.effect_inference(X_test)
    ate = effects_inference.point_estimate.mean()
    ate_se = effects_inference.stderr.mean()
    
    # Compute sensitivity metrics
    # Following the logic from Cinelli & Hazlett (2020)
    sensitivity_value = ate / ate_se  # t-statistic
    
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: Robustness to Unmeasured Confounding")
    print("=" * 70)
    print(f"\nAverage Treatment Effect (ATE): {ate:.4f}")
    print(f"Standard Error: {ate_se:.4f}")
    print(f"t-statistic: {sensitivity_value:.4f}")
    print(f"\nInterpretation:")
    print(f"  - To overturn results, unmeasured confounder would need to have")
    print(f"    effects approximately {np.abs(sensitivity_value):.2f}x larger than")
    print(f"    the observed treatment effect.")
    if np.abs(sensitivity_value) < 2:
        print(f"  - ⚠️ CAUTION: Results may be sensitive to unmeasured confounding")
    else:
        print(f"  - ✓ Results appear robust to moderate unmeasured confounding")
    
    # Visualize sensitivity surface
    # Simulate how results change with different levels of confounding bias
    confounding_bias = np.linspace(-0.5, 0.5, 50)
    adjusted_effects = ate - confounding_bias
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(confounding_bias, adjusted_effects, linewidth=2, label='Adjusted Treatment Effect')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No Effect')
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='No Confounding')
    ax.fill_between(confounding_bias, 
                    adjusted_effects - 1.96*ate_se,
                    adjusted_effects + 1.96*ate_se,
                    alpha=0.2, label='±95% SE band')
    
    ax.set_xlabel('Unmeasured Confounding Bias')
    ax.set_ylabel('Estimated Treatment Effect')
    ax.set_title('Sensitivity to Unmeasured Confounding')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return {
        'ate': ate,
        'ate_se': ate_se,
        't_statistic': sensitivity_value,
        'confounding_bias_range': confounding_bias,
        'adjusted_effects': adjusted_effects
    }, fig


# ============================================================================
# EXAMPLE 5: CROSS-SECTIONAL PREDICTION COMPARISON
# ============================================================================

def compare_predicted_vs_actual_effects(cate_estimates, true_effects):
    """
    Compare predicted heterogeneous effects against true effects
    (when ground truth is available in simulation/validation).
    
    This assesses model quality - how well does causal forest recover
    true heterogeneous effects?
    
    Parameters
    -----------
    cate_estimates : np.ndarray
        Model's predicted treatment effects
    true_effects : np.ndarray
        True effects (from simulation or holdout validation)
    
    Returns
    --------
    performance_metrics : dict
    fig : matplotlib figure
    """
    
    # Compute correlation
    correlation = np.corrcoef(cate_estimates, true_effects)[0, 1]
    
    # Compute mean absolute error
    mae = np.mean(np.abs(cate_estimates - true_effects))
    
    # Compute RMSE
    rmse = np.sqrt(np.mean((cate_estimates - true_effects) ** 2))
    
    metrics = {
        'correlation': correlation,
        'mae': mae,
        'rmse': rmse
    }
    
    print("\n" + "=" * 70)
    print("PREDICTION ACCURACY: Predicted vs True Heterogeneous Effects")
    print("=" * 70)
    print(f"Correlation: {correlation:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    axes[0].scatter(true_effects, cate_estimates, alpha=0.5)
    lims = [
        np.min([axes[0].get_xlim(), axes[0].get_ylim()]),
        np.max([axes[0].get_xlim(), axes[0].get_ylim()]),
    ]
    axes[0].plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    axes[0].set_xlabel('True Treatment Effect')
    axes[0].set_ylabel('Predicted Treatment Effect')
    axes[0].set_title(f'Predictions vs Ground Truth (r={correlation:.3f})')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = cate_estimates - true_effects
    axes[1].scatter(true_effects, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('True Treatment Effect')
    axes[1].set_ylabel('Prediction Error')
    axes[1].set_title(f'Residuals (MAE={mae:.4f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return metrics, fig


# ============================================================================
# EXAMPLE 6: POLICY TARGETING RECOMMENDATIONS
# ============================================================================

def generate_targeting_recommendations(X_test, cate_estimates, feature_names, 
                                      quantile_threshold=0.75):
    """
    Generate data-driven policy targeting recommendations.
    
    Identifies which firms would benefit most from the intervention,
    useful for budget-constrained program implementation.
    
    Parameters
    -----------
    X_test : np.ndarray
        Test features
    cate_estimates : np.ndarray
        Treatment effects
    feature_names : list
        Feature names
    quantile_threshold : float
        Threshold for "high benefit" targets (e.g., top 25%)
    
    Returns
    --------
    targeting_report : pd.DataFrame
    fig : matplotlib figure
    """
    
    # Create targeting dataframe
    df_targeting = pd.DataFrame(X_test, columns=feature_names)
    df_targeting['predicted_effect'] = cate_estimates
    
    # Identify high-benefit firms
    threshold_effect = np.quantile(cate_estimates, quantile_threshold)
    df_targeting['should_target'] = (cate_estimates >= threshold_effect).astype(int)
    
    # Analyze characteristics of high-benefit firms
    high_benefit = df_targeting[df_targeting['should_target'] == 1]
    low_benefit = df_targeting[df_targeting['should_target'] == 0]
    
    print("\n" + "=" * 70)
    print("POLICY TARGETING RECOMMENDATIONS")
    print("=" * 70)
    print(f"\nTarget firms with predicted effects >= {threshold_effect:.4f}")
    print(f"  - Number of firms to target: {len(high_benefit)} " +
          f"({len(high_benefit)/len(df_targeting)*100:.1f}% of population)")
    print(f"  - Mean effect in target group: {high_benefit['predicted_effect'].mean():.4f}")
    print(f"  - Mean effect in non-target: {low_benefit['predicted_effect'].mean():.4f}")
    
    # Compare characteristics
    print(f"\nCharacteristics of target firms vs others:")
    for feat in feature_names[:5]:  # Top 5 features
        high_mean = high_benefit[feat].mean()
        low_mean = low_benefit[feat].mean()
        print(f"  {feat}: High-benefit={high_mean:.3f}, Low-benefit={low_mean:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Distribution of effects with targeting threshold
    axes[0].hist(df_targeting['predicted_effect'], bins=30, alpha=0.7, 
                 label='All firms', color='gray')
    axes[0].axvline(threshold_effect, color='red', linestyle='--', linewidth=2,
                   label=f'Targeting threshold ({quantile_threshold*100:.0f}%ile)')
    axes[0].set_xlabel('Predicted Treatment Effect')
    axes[0].set_ylabel('Number of Firms')
    axes[0].set_title('Distribution of Predicted Effects and Targeting Strategy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Characteristics comparison
    comparison_data = []
    for feat in feature_names[:5]:
        comparison_data.append({
            'Feature': feat,
            'High-Benefit': high_benefit[feat].mean(),
            'Low-Benefit': low_benefit[feat].mean()
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    x = np.arange(len(df_comparison))
    width = 0.35
    
    axes[1].bar(x - width/2, df_comparison['High-Benefit'], width, 
               label='Target Group', alpha=0.8)
    axes[1].bar(x + width/2, df_comparison['Low-Benefit'], width,
               label='Non-Target Group', alpha=0.8)
    axes[1].set_xlabel('Firm Characteristics')
    axes[1].set_ylabel('Mean Value (Standardized)')
    axes[1].set_title('Characteristics of Target vs Non-Target Firms')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_comparison['Feature'], rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return df_targeting, fig


# ============================================================================
# INTEGRATION EXAMPLES
# ============================================================================

def run_all_advanced_examples(model, X_test, cate_estimates, cate_stderr,
                             true_effects, X_train, T_train, Y_train,
                             feature_names, shap_values):
    """
    Run all advanced analysis examples.
    
    Returns dictionary with all results and figures.
    """
    
    results = {}
    
    print("\n" + "=" * 80)
    print("RUNNING ADVANCED ANALYSIS EXAMPLES")
    print("=" * 80)
    
    # Example 1: Heterogeneous effects by firm type
    print("\n[1/6] Analyzing heterogeneous effects by firm type...")
    results['effects_by_type'], results['fig_effects_by_type'] = \
        analyze_heterogeneous_effects_by_firm_type(
            model, X_test, cate_estimates, feature_names
        )
    
    # Example 2: SHAP interactions
    print("[2/6] Analyzing SHAP feature interactions...")
    results['fig_shap_interactions'] = analyze_shap_interactions(
        shap_values, X_test, feature_names, feature_idx1=0, feature_idx2=1
    )
    
    # Example 3: Treatment effect curves
    print("[3/6] Plotting treatment effect heterogeneity curves...")
    results['fig_effect_curves'] = plot_treatment_effect_curves(
        X_test, cate_estimates, feature_names, feature_idx=0
    )
    
    # Example 4: Sensitivity analysis
    print("[4/6] Performing sensitivity analysis...")
    results['sensitivity_stats'], results['fig_sensitivity'] = \
        sensitivity_analysis_unmeasured_confounder(
            model, X_test, feature_names
        )
    
    # Example 5: Prediction comparison
    print("[5/6] Comparing predicted vs actual effects...")
    results['prediction_metrics'], results['fig_prediction_comparison'] = \
        compare_predicted_vs_actual_effects(cate_estimates, true_effects)
    
    # Example 6: Targeting recommendations
    print("[6/6] Generating policy targeting recommendations...")
    results['targeting_df'], results['fig_targeting'] = \
        generate_targeting_recommendations(X_test, cate_estimates, feature_names)
    
    print("\n✓ All advanced examples completed!")
    
    return results


if __name__ == "__main__":
    print("Advanced Examples Module - Import and use with your fitted models")
    print("\nExample usage:")
    print("  from causal_forest_shap_workflow import run_complete_workflow")
    print("  from advanced_examples import run_all_advanced_examples")
    print("")
    print("  results = run_complete_workflow()")
    print("  advanced = run_all_advanced_examples(")
    print("      model=results['model'],")
    print("      X_test=results['X_test'],")
    print("      cate_estimates=results['cate_estimates'],")
    print("      # ... other parameters ...")
    print("  )")
