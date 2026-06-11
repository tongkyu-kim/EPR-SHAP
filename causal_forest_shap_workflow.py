"""
Complete Workflow: CausalForestDML + SHAP Integration for Heterogeneous Treatment Effects

This module demonstrates a production-ready implementation of:
1. CausalForestDML for heterogeneous treatment effect (CATE) estimation
2. SHAP integration with causal forests for treatment effect interpretation
3. Complete workflow from raw data to SHAP visualizations

Key Features:
- Robust handling of firm-level covariates
- Statistical inference on treatment effects
- Multiple SHAP visualization types (beeswarm, dependence, force plots)
- Sensitivity analysis and feature importance extraction

Sources:
- EconML Documentation: https://www.pywhy.org/EconML/
- EconML Causal Forest Examples: https://github.com/py-why/EconML/blob/main/notebooks/Causal%20Forest%20and%20Orthogonal%20Random%20Forest%20Examples.ipynb
- SHAP Documentation: https://shap.readthedocs.io/
- EconML SHAP Integration: https://github.com/py-why/EconML/blob/main/notebooks/Interpretability%20with%20SHAP.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import shap

# Import EconML components
from econml.dml import CausalForestDML
from econml.iv.dr import ForestDRIV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, LassoCV


# ============================================================================
# PART 1: DATA GENERATION AND PREPROCESSING
# ============================================================================

def generate_treatment_effect_data(n_samples=1000, n_features=10, effect_type='heterogeneous'):
    """
    Generate synthetic experimental data with heterogeneous treatment effects.
    
    Parameters
    -----------
    n_samples : int
        Number of observations (firms or units)
    n_features : int
        Number of covariates/firm characteristics
    effect_type : str
        Type of treatment effect: 'homogeneous' or 'heterogeneous'
    
    Returns
    --------
    X : np.ndarray
        Covariates (firm characteristics) - shape (n_samples, n_features)
    T : np.ndarray
        Binary treatment indicator (e.g., program participation) - shape (n_samples,)
    Y : np.ndarray
        Outcome variable (e.g., firm performance) - shape (n_samples,)
    true_effects : np.ndarray
        True causal effects (used for validation) - shape (n_samples,)
    feature_names : list
        Names of features for interpretation
    
    Notes
    ------
    Simulates firm-level data where:
    - X contains firm characteristics (size, R&D intensity, market share, etc.)
    - T is binary treatment (policy intervention, program participation)
    - Y is firm outcome (productivity, export, innovation, etc.)
    - Treatment effects vary with firm characteristics (heterogeneous)
    """
    np.random.seed(42)
    
    # Generate firm characteristics (covariates)
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Feature names for interpretation
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    feature_names[0] = 'Firm_Size'
    feature_names[1] = 'R&D_Intensity'
    feature_names[2] = 'Market_Share'
    
    # Treatment assignment: propensity score depends on covariates
    # This creates confoundedness (common scenario in observational data)
    propensity_score = 0.3 + 0.2 * X[:, 0] + 0.15 * X[:, 1]
    propensity_score = np.clip(propensity_score, 0.1, 0.9)
    T = np.random.binomial(1, propensity_score)
    
    # Generate outcome
    # Base outcome (outcome under control)
    base_outcome = 2.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2]
    
    # Treatment effect - HETEROGENEOUS in firm characteristics
    if effect_type == 'heterogeneous':
        # Treatment effect increases with firm size and R&D intensity
        true_effects = (1.0 +                    # Base treatment effect
                       0.5 * X[:, 0] +          # Effect increases with firm size
                       0.3 * X[:, 1] -          # Effect increases with R&D
                       0.2 * X[:, 0] * X[:, 1])  # Interaction effects
    else:
        # Homogeneous effect (constant across firms)
        true_effects = np.ones(n_samples) * 1.5
    
    # Add random noise to outcomes
    noise = np.random.normal(0, 0.5, size=n_samples)
    Y = base_outcome + T * true_effects + noise
    
    return X, T, Y, true_effects, feature_names


# ============================================================================
# PART 2: CAUSAL FOREST DML ESTIMATION
# ============================================================================

def fit_causal_forest_dml(X, T, Y, test_size=0.2, random_state=42):
    """
    Fit CausalForestDML model for heterogeneous treatment effect estimation.
    
    CausalForestDML implements:
    - Double Machine Learning (DML) to deconfound treatment assignment
    - Forest-based CATE (Conditional Average Treatment Effect) estimation
    - Provides both point estimates and confidence intervals
    
    Parameters
    -----------
    X : np.ndarray
        Feature matrix (covariates)
    T : np.ndarray
        Treatment vector (binary or continuous)
    Y : np.ndarray
        Outcome vector
    test_size : float
        Fraction of data for final model training (typical: 0.2-0.3)
    random_state : int
        Seed for reproducibility
    
    Returns
    --------
    model : CausalForestDML
        Fitted causal forest model
    X_train, X_test, T_train, T_test, Y_train, Y_test : arrays
        Train/test split data for later validation
    
    Implementation Details
    -----------------------
    The CausalForestDML process:
    1. First stage: Fit nuisance models (propensity score, outcome model)
    2. Deconfounding: Compute residuals to remove confounding effects
    3. Second stage: Fit causal forest on deconfounded residuals
    4. CATE estimation: Predict heterogeneous treatment effects
    
    Key Parameters
    ---------------
    n_trees : int
        Number of trees in forest (default 100, increase for larger data)
    max_depth : int
        Maximum tree depth (typical 15-30)
    min_samples_leaf : int
        Minimum samples per leaf for honest splits (typical 5-10)
    honest : bool
        Use honest splitting for valid statistical inference
    
    References
    -----------
    - Athey, S., & Wager, S. (2019). Generalized random forests
    - EconML: https://www.pywhy.org/EconML/spec/estimation/forest.html
    """
    
    # Split data for sample efficiency
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=test_size, random_state=random_state
    )
    
    # Initialize nuisance models (first stage models)
    # These estimate propensity score p(T|X) and outcome E[Y|X,T]
    model_y = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=random_state
    )
    
    model_t = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=random_state
    )
    
    # Initialize CausalForestDML
    # This is the main estimator for heterogeneous treatment effects
    est = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        model_final=None,  # Forest handles final stage automatically
        cv=5,  # Cross-validation folds for nuisance estimation
        n_estimators=500,  # Number of trees in causal forest
        max_depth=15,  # Maximum depth for honest forest
        min_samples_leaf=5,  # Minimum samples for honest inference
        random_state=random_state,
        verbose=0
    )
    
    # Fit the model on training data
    print("Fitting CausalForestDML on training data...")
    est.fit(Y_train, T_train, X=X_train)
    
    print("Model fitted successfully!")
    print(f"  - Average Treatment Effect (ATE): {est.ate(X_test):.4f}")
    print(f"  - ATE Standard Error: {est.ate_inference(X_test).stderr[0]:.4f}")
    
    return est, X_train, X_test, T_train, T_test, Y_train, Y_test


def estimate_conditional_treatment_effects(model, X, X_test):
    """
    Estimate Conditional Average Treatment Effects (CATE) for samples.
    
    CATE(X) = E[Y(1) - Y(0) | X]
    The heterogeneous treatment effect for each unit given their covariates.
    
    Parameters
    -----------
    model : CausalForestDML
        Fitted causal forest model
    X : np.ndarray
        Training data (for reference)
    X_test : np.ndarray
        Test samples where to predict treatment effects
    
    Returns
    --------
    cate_estimates : np.ndarray
        Conditional Average Treatment Effects - shape (n_test,)
    cate_stderr : np.ndarray
        Standard errors of CATE estimates
    cate_ci : tuple
        95% confidence intervals for CATE
    
    Output Interpretation
    ----------------------
    - CATE > 0: Treatment has positive effect for this unit
    - CATE < 0: Treatment has negative effect for this unit
    - |CATE| > stderr*1.96: Effect is statistically significant at 5% level
    """
    
    # Get point estimates of heterogeneous effects
    cate_estimates = model.effect(X_test)
    
    # Get inference objects for confidence intervals
    cate_inf = model.effect_inference(X_test)
    cate_stderr = cate_inf.stderr
    
    # Compute 95% confidence intervals
    cate_ci = cate_inf.conf_int(alpha=0.05)
    
    print(f"\nConditional Average Treatment Effects (CATE):")
    print(f"  - Mean CATE: {cate_estimates.mean():.4f}")
    print(f"  - Std Dev of CATE: {cate_estimates.std():.4f}")
    print(f"  - Range: [{cate_estimates.min():.4f}, {cate_estimates.max():.4f}]")
    
    return cate_estimates, cate_stderr, cate_ci


# ============================================================================
# PART 3: SHAP INTEGRATION FOR CAUSAL FOREST INTERPRETATION
# ============================================================================

def extract_shap_values_from_causal_forest(model, X_test, n_samples=100):
    """
    Extract SHAP values for causal forest treatment effect predictions.
    
    SHAP (SHapley Additive exPlanations) values decompose the prediction
    for each sample, showing the contribution of each feature.
    
    For causal forests predicting treatment effects:
    CATE(X) ≈ base_value + Σ(SHAP_value_i * (X_i - E[X_i]))
    
    Parameters
    -----------
    model : CausalForestDML
        Fitted causal forest model
    X_test : np.ndarray
        Test samples for which to compute SHAP values
    n_samples : int
        Number of samples to use for TreeSHAP background (typically 100-300)
    
    Returns
    --------
    shap_values : np.ndarray
        SHAP values for each feature - shape (n_test, n_features)
    base_value : float
        Model's base prediction (average prediction)
    
    Notes
    ------
    For forest-based models, we use TreeSHAP which is fast and provides
    exact Shapley values directly from tree structure.
    
    Interpretation:
    - Positive SHAP: Feature increases treatment effect
    - Negative SHAP: Feature decreases treatment effect
    - Magnitude indicates importance of contribution
    
    References
    -----------
    - SHAP TreeSHAP: https://arxiv.org/abs/1905.04610
    - SHAP Documentation: https://shap.readthedocs.io/
    """
    
    print("\nExtracting SHAP values from causal forest...")
    
    # Get the underlying forest model from CausalForestDML
    # The forest is stored in the model_cate attribute
    forest_model = model.model_cate
    
    # Create SHAP explainer for forest model
    # For forests, TreeSHAP provides exact Shapley values
    explainer = shap.TreeExplainer(forest_model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_test)
    base_value = explainer.expected_value
    
    print(f"SHAP values computed:")
    print(f"  - Shape: {shap_values.shape}")
    print(f"  - Base value (avg prediction): {base_value:.4f}")
    print(f"  - Mean absolute SHAP: {np.abs(shap_values).mean(axis=0)}")
    
    return shap_values, base_value, explainer


# ============================================================================
# PART 4: SHAP VISUALIZATION FUNCTIONS
# ============================================================================

def plot_shap_summary_beeswarm(shap_values, X_test, feature_names, max_display=15):
    """
    Create SHAP beeswarm plot: Feature importance with distribution.
    
    Shows:
    - Which features are most important for predictions
    - How feature values affect predictions
    - Direction of effects (color: red=high, blue=low feature values)
    
    Parameters
    -----------
    shap_values : np.ndarray
        SHAP values for test samples - shape (n_test, n_features)
    X_test : np.ndarray
        Feature values for test samples
    feature_names : list
        Names of features for labels
    max_display : int
        Maximum features to display
    
    Returns
    --------
    fig : matplotlib figure
    
    Interpretation
    ---------------
    - Width shows range of SHAP values
    - Color shows feature value magnitude
    - Red dots (right): features with high values increase prediction
    - Blue dots (left): features with low values increase prediction
    """
    
    # Create dataframe for SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot summary
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type="beeswarm", max_display=max_display, show=False)
    
    plt.title("SHAP Summary Plot: Feature Importance for Treatment Effects")
    plt.xlabel("SHAP value (Impact on treatment effect prediction)")
    plt.tight_layout()
    
    return plt.gcf()


def plot_shap_dependence_plots(shap_values, X_test, feature_names, 
                               feature_indices=None, figsize=(15, 10)):
    """
    Create SHAP dependence plots for selected features.
    
    Shows the relationship between feature value and SHAP contribution,
    automatically colored by interacting feature.
    
    Parameters
    -----------
    shap_values : np.ndarray
        SHAP values for samples
    X_test : np.ndarray
        Feature values
    feature_names : list
        Feature names
    feature_indices : list or None
        Indices of features to plot (None = top features)
    figsize : tuple
        Figure size
    
    Returns
    --------
    fig : matplotlib figure
    
    Interpretation
    ---------------
    - X-axis: Feature value
    - Y-axis: SHAP value (impact on prediction)
    - Color: Value of interacting feature (often automatically detected)
    - Shape shows how feature affects treatment effect prediction
    """
    
    if feature_indices is None:
        # Plot top features by average absolute SHAP value
        feature_importance = np.abs(shap_values).mean(axis=0)
        feature_indices = np.argsort(feature_importance)[-6:][::-1]
    
    n_features = len(feature_indices)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature_idx in enumerate(feature_indices):
        ax = axes[idx]
        
        # Create dependence plot
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X_test,
            feature_names=feature_names,
            ax=ax,
            show=False
        )
        
        ax.set_title(f"SHAP Dependence: {feature_names[feature_idx]}")
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_shap_force_plots(shap_values, X_test, feature_names, 
                          sample_indices=None, n_samples=3):
    """
    Create SHAP force plots: Waterfall charts of predictions.
    
    Shows how each feature contributes (pushes) the prediction
    away from the base value for individual samples.
    
    Parameters
    -----------
    shap_values : np.ndarray
        SHAP values
    X_test : np.ndarray
        Feature values
    feature_names : list
        Feature names
    sample_indices : list or None
        Specific samples to visualize (None = random samples)
    n_samples : int
        Number of samples to plot if sample_indices is None
    
    Returns
    --------
    list of fig : matplotlib figures
    
    Interpretation
    ---------------
    - Red arrows: Features pushing prediction up (positive effects)
    - Blue arrows: Features pushing prediction down (negative effects)
    - Length of arrow: Magnitude of feature's contribution
    - Starts at base value, ends at model prediction
    """
    
    if sample_indices is None:
        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    figs = []
    for sample_idx in sample_indices:
        # Create force plot
        fig = shap.force_plot(
            0,  # base_value
            shap_values[sample_idx],
            X_test[sample_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        figs.append(fig)
    
    return figs


def plot_feature_importance_from_shap(shap_values, feature_names):
    """
    Plot mean absolute SHAP values as feature importance.
    
    Simple bar chart showing which features are most important
    for the model's treatment effect predictions.
    
    Parameters
    -----------
    shap_values : np.ndarray
        SHAP values for all samples
    feature_names : list
        Feature names
    
    Returns
    --------
    fig : matplotlib figure
    """
    
    # Compute mean absolute SHAP values
    importance = np.abs(shap_values).mean(axis=0)
    
    # Sort by importance
    sorted_idx = np.argsort(importance)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(sorted_idx))
    ax.barh(y_pos, importance[sorted_idx], color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance for Treatment Effect Predictions")
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


# ============================================================================
# PART 5: COMPLETE WORKFLOW
# ============================================================================

def run_complete_workflow(n_samples=500, n_features=10, random_state=42):
    """
    Execute complete workflow: Data → CausalForest → SHAP → Visualizations
    
    This is the main entry point that demonstrates the full pipeline
    for causal effect estimation and SHAP-based interpretation.
    
    Returns
    --------
    results : dict
        Contains all results, models, and data
    """
    
    print("=" * 80)
    print("COMPLETE WORKFLOW: CausalForestDML + SHAP Integration")
    print("=" * 80)
    
    # ========================================================================
    # Step 1: Generate or Load Data
    # ========================================================================
    print("\n[STEP 1] Generating experimental data...")
    X, T, Y, true_effects, feature_names = generate_treatment_effect_data(
        n_samples=n_samples,
        n_features=n_features,
        effect_type='heterogeneous'
    )
    
    print(f"  - Data shape: X={X.shape}, T={T.shape}, Y={Y.shape}")
    print(f"  - Features: {feature_names[:5]}... ({n_features} total)")
    print(f"  - Treatment rate: {T.mean():.1%}")
    
    # ========================================================================
    # Step 2: Fit CausalForestDML
    # ========================================================================
    print("\n[STEP 2] Fitting CausalForestDML model...")
    model, X_train, X_test, T_train, T_test, Y_train, Y_test = \
        fit_causal_forest_dml(X, T, Y, test_size=0.2, random_state=random_state)
    
    # ========================================================================
    # Step 3: Estimate Conditional Treatment Effects (CATE)
    # ========================================================================
    print("\n[STEP 3] Estimating heterogeneous treatment effects...")
    cate_estimates, cate_stderr, cate_ci = estimate_conditional_treatment_effects(
        model, X, X_test
    )
    
    # ========================================================================
    # Step 4: Extract SHAP Values
    # ========================================================================
    print("\n[STEP 4] Computing SHAP values for interpretation...")
    shap_values, base_value, explainer = extract_shap_values_from_causal_forest(
        model, X_test, n_samples=100
    )
    
    # ========================================================================
    # Step 5: Create SHAP Visualizations
    # ========================================================================
    print("\n[STEP 5] Creating SHAP visualizations...")
    
    # Create results dictionary
    results = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'T_train': T_train,
        'T_test': T_test,
        'Y_train': Y_train,
        'Y_test': Y_test,
        'cate_estimates': cate_estimates,
        'cate_stderr': cate_stderr,
        'cate_ci': cate_ci,
        'true_effects': true_effects,
        'shap_values': shap_values,
        'base_value': base_value,
        'explainer': explainer,
        'feature_names': feature_names,
        'figures': {}
    }
    
    # Create visualizations
    print("  - Creating beeswarm plot...")
    results['figures']['beeswarm'] = plot_shap_summary_beeswarm(
        shap_values, X_test, feature_names
    )
    
    print("  - Creating dependence plots...")
    results['figures']['dependence'] = plot_shap_dependence_plots(
        shap_values, X_test, feature_names
    )
    
    print("  - Creating force plots...")
    results['figures']['force'] = plot_shap_force_plots(
        shap_values, X_test, feature_names, n_samples=3
    )
    
    print("  - Creating feature importance plot...")
    results['figures']['importance'] = plot_feature_importance_from_shap(
        shap_values, feature_names
    )
    
    # ========================================================================
    # Step 6: Summary and Diagnostics
    # ========================================================================
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE - Summary Statistics")
    print("=" * 80)
    
    print(f"\nCausal Effects:")
    print(f"  - ATE (Average Treatment Effect): {cate_estimates.mean():.4f}")
    print(f"  - Std Dev of CATE: {cate_estimates.std():.4f}")
    print(f"  - Min CATE: {cate_estimates.min():.4f}")
    print(f"  - Max CATE: {cate_estimates.max():.4f}")
    
    print(f"\nSHAP Analysis:")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-3:][::-1]
    print(f"  - Top 3 most important features:")
    for i, feat_idx in enumerate(top_features_idx):
        print(f"    {i+1}. {feature_names[feat_idx]}: {mean_abs_shap[feat_idx]:.4f}")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run complete workflow
    results = run_complete_workflow(n_samples=500, n_features=10, random_state=42)
    
    # Save results for further analysis
    print("\nWorkflow execution complete!")
    print("Results dictionary contains:")
    print(f"  - model: Fitted CausalForestDML")
    print(f"  - cate_estimates: Heterogeneous treatment effects")
    print(f"  - shap_values: SHAP explanations")
    print(f"  - figures: Matplotlib figures for visualization")
