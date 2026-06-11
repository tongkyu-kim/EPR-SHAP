# EPR-SHAP

## Governance Mechanisms and Extended Producer Responsibility (EPR) Compliance

This repository contains code, data pipelines, and analytical workflows for a randomized survey experiment examining how different governance mechanisms influence firms' willingness to comply with Extended Producer Responsibility (EPR) regulations.

The project combines causal inference and explainable machine learning to investigate not only whether governance interventions work, but also **which governance mechanisms work for which firms and why**.

---

## Research Question

Why do firms respond differently to environmental governance mechanisms?

Specifically, this project compares the effectiveness of three competing governance mechanisms:

* **Regulatory Pressure** — inspections, audits, monitoring, and penalties
* **Reputational Pressure** — consumer scrutiny, stakeholder expectations, and corporate image
* **Market Pressure** — buyer requirements, supply-chain standards, and export market demands

The central premise is that governance effectiveness is contingent upon firm characteristics, organizational capabilities, and supply-chain positioning.

---

## Analytical Framework

```text
Randomized Survey Experiment
            ↓
Average Treatment Effects (ATE)
            ↓
Conditional Treatment Effects (CATE)
            ↓
Causal Forests (EconML)
            ↓
SHAP Interpretation
            ↓
Governance Response Profiles
```

Rather than focusing solely on average treatment effects, the analysis seeks to identify heterogeneous responses across firms and uncover the organizational characteristics that drive governance effectiveness.

---

## Repository Structure

```text
EPR-SHAP/
├── data/
│   └── processed/
│       └── survey_data_clean.csv
│
├── outputs/
│   └── governance/
│       ├── tables/
│       ├── figures/
│       └── shap/
│
├── src/
│   ├── preprocessing/
│   ├── ate/
│   ├── cate/
│   ├── governance/
│   └── shap_analysis/
│
├── survey/
│   └── codebook/
│
├── causal_forest_shap_workflow.py
├── run_governance_analysis.py
├── advanced_examples.py
├── generate_report.py
├── quick_start.py
└── requirements.txt
```

---

## Key Outputs

* Randomization and balance checks
* Treatment effect estimates (ATE)
* Heterogeneous treatment effects (CATE)
* Governance response profiles
* SHAP feature importance analysis
* SHAP dependence plots
* Publication-ready tables and figures

---

## Research Objectives

* Evaluate competing theories of environmental compliance
* Identify heterogeneous responses to governance interventions
* Explain treatment-effect heterogeneity using SHAP
* Develop evidence for targeted and adaptive EPR governance strategies
* Contribute to research on circular economy governance and environmental policy design

---

## Target Journals

* Resources, Conservation & Recycling (RCR)
* Sustainable Production and Consumption (SPC)
* Business Strategy and the Environment (BSE)
* Technological Forecasting and Social Change (TFSC)

```
```
