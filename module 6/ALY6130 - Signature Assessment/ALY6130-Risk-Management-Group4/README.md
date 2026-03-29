# ALY6130 Risk Management Analytics — Group 4

**ServiceNow's $7.75 Billion Acquisition of Armis: Enterprise Risk Assessment**

Northeastern University Vancouver | College of Professional Studies | March 2026  
Group 4: Adwoa Bempomaa, Sara Mathai, Lawrence Dass

---

## Overview

Full enterprise risk assessment of ServiceNow's acquisition of Armis across 25 identified risks spanning six categories: Strategic, Operational, Financial, Technical, Regulatory, and Market.

**Key quantitative findings:**
- Net expected annual portfolio impact: **-$163.7M** (81.3% probability of net loss)
- 90% confidence interval: **-$470.5M to +$131.9M**
- ML escalation probability (current state): **58.2%**
- Only Red KRI: **R24** (Multi-Cloud Redundancy at 42%)
- Only Green KRI: **R3** (Regulatory Pipeline Growth at 32% QoQ)

---

## Repository Structure

```
ALY6130-Risk-Management-Group4/
├── enterprise_risk_analysis.py        # Master Python script (all 8 steps)
├── requirements.txt                   # Python dependencies
├── data/
│   ├── raw/                           # Raw input data (not applicable — synthetic)
│   └── processed/
│       ├── ALY6130_Signature_Assessment_Group4.xlsx  # Excel workbook (11 sheets)
│       ├── analytical_metrics.csv                    # Expected values, 25 risks
│       ├── category_portfolios.csv                   # Category-level MC results
│       ├── current_portfolio_state.csv               # Current KRI readings
│       ├── executive_summary.txt                     # Key output summary
│       ├── feature_importance.csv                    # ML permutation importance
│       ├── ml_training_data.csv                      # 500-obs synthetic dataset
│       ├── monte_carlo_summary.csv                   # Per-risk MC summary
│       └── portfolio_key_stats.csv                   # Locked portfolio statistics
├── notebooks/
│   ├── eda.ipynb                      # Exploratory data analysis
│   ├── qualitative_analysis.ipynb     # Scenario + Industry Fusion Analytics
│   ├── monte_carlo.ipynb              # Monte Carlo simulation
│   └── quantitative_analysis.ipynb   # ML model, feature importance, scoring
├── report/
│   ├── Final_Signature_Assessment_Group4_WithFigures.docx  # Final report
│   ├── figure1_risk_heatmap.png
│   ├── figure2_mc_net_distribution.png
│   ├── figure3_individual_distributions.png
│   ├── figure4_category_portfolios.png
│   ├── figure5_feature_importance.png
│   ├── figure6_kri_dashboard.png
│   ├── figure7_confusion_matrix.png
│   └── figure8_scenario_analysis.png
└── scripts/                           # Reserved for future utility scripts
```

---

## Reproducibility

All quantitative outputs are reproducible with `seed=42`:

```bash
pip install -r requirements.txt
python enterprise_risk_analysis.py
```

The script executes 8 sequential steps:
1. Analytical expected values (Bernoulli-Triangular model)
2. Monte Carlo simulation (10,000 iterations, seed=42)
3. Category-level portfolio analysis (6 categories)
4. Synthetic dataset generation (500 observations, 15 features)
5. ML model training and cross-validation (Random Forest, 5-fold CV)
6. Current portfolio state scoring
7. All 8 report figures
8. Executive summary

---

## Excel Workbook (11 Sheets)

| Sheet | Contents |
|-------|----------|
| Cover | Workbook guide and sheet index |
| RiskRegister_Quantitative | All 25 risks: formal three-part statements + quantitative parameters |
| Risk_Heatmap | Likelihood-Impact grid plotting all 25 risks |
| MC_Parameters | Monte Carlo parameters and locked portfolio results |
| IW_Framework | KRI thresholds (Green/Amber/Red) for all 25 risks |
| Summary_Statistics | Portfolio statistics, category breakdown, ML model results |
| Risk_Mapping | Risk-strategy-owner cross-reference |
| RiskCalculationSheet | 1-9 probability and impact scoring rubric |
| M2_ProblemStatement | Module 2 reference: original 3-risk problem statements |
| M2_KRI_Framework | Module 2 reference: original 3-risk KRI framework |
| M2_RiskRegister | Module 2 reference: original 3-risk register |

---

## Risk Register Summary (25 Risks)

| Priority | Count | Key Categories |
|----------|-------|----------------|
| HIGH | 8 | Strategic (5), Regulatory (2), Market (1) |
| MEDIUM | 14 | Operational (4), Technical (4), Financial (2), Regulatory (2), Strategic (1), Market (1) |
| LOW | 3 | Operational (1), Market (1), Technical (1) |

**Positive risks:** R3 (Regulatory Demand), R19 (OT/IoT Market Growth), R20 (Cross-Sell Revenue)  
**Highest impact risk:** R2 (AI Platform Convergence, impact score 9)  
**Highest score risk:** R3 (Regulatory Demand Acceleration, score 72)

---

## Key References

- Armis. (2025a). Armis to join ServiceNow. Armis Blog.
- Armis. (2025b). Armis surpasses $300M ARR. Armis Newsroom.
- Fleisher, C. S., & Bensoussan, B. E. (2015). *Business and competitive analysis* (2nd ed.). FT Press.
- Forrester Research. (2025). The state of cybersecurity spending 2025.
- Gartner. (2025). Forecast: Information security, worldwide, 2023-2028.
- Jalilvand, A., & Moorthy, S. (2023). Triangulating risk profile and risk assessment. *Journal of Risk and Financial Management*, 16(11), 473.
- MarketsandMarkets. (2025). OT security market global forecast to 2030.
- Palmer, A. (2025, December 23). ServiceNow acquiring Armis for nearly $8 billion. CNBC.
- ServiceNow. (2025a). ServiceNow to acquire Armis. ServiceNow Newsroom.
- ServiceNow. (2026). Q4 and full-year 2025 financial results. ServiceNow IR.
