EPR-SHAP

What Drives Extended Producer Responsibility (EPR) Compliance? Experimental Evidence on Enforcement, Reputation, and Market Pressure

This repository contains code and analysis for a randomized survey experiment examining how different governance mechanisms influence EPR compliance among firms.

Research Question

Which governance mechanisms most effectively promote EPR compliance, and why do different firms respond differently to regulatory, reputational, and market pressures?

The project investigates three competing governance mechanisms:

Regulatory Pressure (inspections, audits, penalties)
Reputational Pressure (consumer scrutiny, corporate image)
Market Pressure (buyer requirements, supply chain expectations)
Methodology

The analytical workflow combines causal inference and explainable AI:

Randomized Survey Experiment
Average Treatment Effect (ATE) Estimation
Heterogeneous Treatment Effect Analysis (CATE)
Causal Forests (EconML)
SHAP-based Interpretation of Treatment Heterogeneity

Rather than focusing solely on average treatment effects, the project examines which governance mechanisms work for which firms and identifies the organizational characteristics associated with differential responsiveness.

Workflow
Survey Experiment
        ↓
ATE Estimation
        ↓
Causal Forests
        ↓
CATE Estimation
        ↓
SHAP Analysis
        ↓
Governance Response Profiles
Repository Structure
EPR-SHAP/
├── data/
│   └── processed/
│       └── survey_data_clean.csv      # Cleaned firm-level survey data
├── outputs/
│   └── governance/
│       ├── tables/                    # CSV result tables
│       ├── figures/                   # PNG plots
│       └── shap/                      # SHAP outputs (optional)
├── src/
│   ├── preprocessing/
│   ├── ate/
│   ├── cate/
│   ├── governance/
│   └── shap_analysis/
├── survey/
│   └── codebook/                      # Survey instrument and variable descriptions
├── causal_forest_shap_workflow.py     # Core CausalForestDML + SHAP pipeline
├── run_governance_analysis.py         # Full end-to-end analysis (Stages 1–7)
├── run_cate_test.py                   # CATE unit tests
├── run_shap_test.py                   # SHAP unit tests
├── advanced_examples.py               # Extended analyses (subgroups, sensitivity)
├── generate_report.py                 # Automated report generation
├── quick_start.py                     # Minimal usage examples
└── requirements.txt
Key Outputs
Balance and randomization checks
Treatment effect estimates
Heterogeneous treatment effects
SHAP feature importance
SHAP dependence plots
Governance response profiles
Publication-ready tables and figures
