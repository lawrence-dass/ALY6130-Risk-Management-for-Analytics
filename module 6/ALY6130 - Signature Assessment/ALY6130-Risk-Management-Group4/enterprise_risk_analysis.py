"""
Enterprise Risk Analysis: ServiceNow-Armis Acquisition
ALY 6130 - Risk Management Analytics
Group 4: Adwoa Bempomaa, Sara Mathai, Lawrence Dass
March 2026

Purpose
-------
Full enterprise risk assessment across all 25 identified risks from the
Module 3 risk register. Implements:
  1. Standardized quantitative risk modeling (Bernoulli-Triangular)
  2. Monte Carlo simulation (10,000 iterations, seed=42)
  3. ML-based risk escalation prediction (Random Forest, 15 features)
  4. Category-level portfolio analysis
  5. Feature importance aligned with KRI framework
  6. Current portfolio state scoring

Synthetic Data Justification
-----------------------------
The ServiceNow-Armis acquisition has not yet closed (expected H2 2026).
No historical internal escalation data exists for this specific integration.
Synthetic observations are generated using the same KRI logic and threshold
structure developed in the qualitative I&W framework, as explicitly
permitted by the assignment instructions. The label-generation process
encodes realistic business assumptions drawn from the competitive landscape,
regulatory environment, and integration risk profile.

Outputs
-------
All CSV outputs saved to data/processed/
All PNG plots saved to report/
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
SEED = 42
N_ITERATIONS = 10_000
N_SYNTHETIC = 500
np.random.seed(SEED)

ROOT = Path("/home/claude/enterprise-risk-project")
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
REPORT = ROOT / "report"
for d in [DATA_RAW, DATA_PROC, REPORT]:
    d.mkdir(parents=True, exist_ok=True)

# ─── RISK DEFINITIONS ─────────────────────────────────────────────────────────
@dataclass
class Risk:
    risk_id: str
    name: str
    category: str          # Strategic, Operational, Financial, Technical, Regulatory, Market
    risk_type: str         # Positive / Negative
    likelihood: int        # 1-9 ordinal scale
    impact: int            # 1-9 ordinal scale
    risk_score: int        # likelihood * impact
    priority: str          # HIGH / MEDIUM / LOW
    probability: float     # derived from likelihood
    low_m: float           # triangular lower bound ($M)
    mode_m: float          # triangular mode ($M)
    high_m: float          # triangular upper bound ($M)
    sign: int              # +1 upside / -1 downside
    owner: str
    kri_name: str
    green: str
    amber: str
    red: str
    current_kri: str
    kri_status: str        # Green / Amber / Red


# Probability mapping: likelihood score -> probability
def prob(l: int) -> float:
    return {1: 0.10, 3: 0.30, 5: 0.50, 7: 0.70, 9: 0.90}.get(l, l / 10)


# Impact calibration rationale:
# Exposure base = $1B Security & Risk ACV for strategic/revenue risks
# Integration budget proxy = $150M for operational risks
# Regulatory penalty/cost proxy for compliance risks
# Market revenue proxy for market opportunity risks

RISKS: List[Risk] = [
    # ── POSITIVE RISKS ────────────────────────────────────────────────────────
    Risk("R3",  "Regulatory-Driven Demand Acceleration",
         "Regulatory", "Positive", 9, 8, 72, "HIGH",
         prob(9), 60.0, 250.0, 350.0, +1,
         "VP Product / VP Federal Sales",
         "Regulatory Pipeline Growth Rate (QoQ)",
         ">25% QoQ growth", "10-25% QoQ growth", "<10% QoQ growth",
         "32% QoQ", "Green"),

    Risk("R19", "OT/IoT Security Market Growth Opportunity",
         "Market", "Positive", 7, 8, 56, "HIGH",
         prob(7), 30.0, 120.0, 200.0, +1,
         "Chief Product Officer",
         "OT/IoT Market Share Capture Rate",
         ">15% YoY share gain", "5-15% YoY share gain", "<5% YoY share gain",
         "8% YoY", "Amber"),

    Risk("R20", "Cross-Sell Revenue via Existing Customer Base",
         "Market", "Positive", 7, 8, 56, "HIGH",
         prob(7), 25.0, 100.0, 180.0, +1,
         "Chief Revenue Officer",
         "Cross-Sell Adoption Rate",
         ">20% existing customers adopt Armis", "10-20%", "<10%",
         "12%", "Amber"),

    # ── NEGATIVE HIGH ─────────────────────────────────────────────────────────
    Risk("R2",  "AI-Driven Platform Convergence Risk",
         "Strategic", "Negative", 7, 9, 63, "HIGH",
         prob(7), 40.0, 120.0, 200.0, -1,
         "Chief Product Officer / Chief Strategy Officer",
         "AI Capability Parity Index",
         ">=85% feature parity", "70-84% feature parity", "<70% feature parity",
         "78%", "Amber"),

    Risk("R1",  "Competitive Displacement by Rival Mega-Acquisitions",
         "Strategic", "Negative", 7, 8, 56, "HIGH",
         prob(7), 30.0, 100.0, 150.0, -1,
         "Chief Strategy Officer / CRO",
         "Competitive Win Rate in OT/IoT Deals",
         ">35% win rate", "25-35% win rate", "<25% win rate",
         "31%", "Amber"),

    Risk("R13", "AI Technology Obsolescence",
         "Strategic", "Negative", 7, 8, 56, "HIGH",
         prob(7), 25.0, 80.0, 130.0, -1,
         "Chief Product Officer",
         "AI Feature Release Velocity vs Competitors",
         ">90% parity", "75-90% parity", "<75% parity",
         "77%", "Amber"),

    Risk("R17", "Customer Migration to Competitor Platforms",
         "Strategic", "Negative", 7, 8, 56, "HIGH",
         prob(7), 20.0, 75.0, 120.0, -1,
         "Chief Revenue Officer",
         "Customer Retention Rate",
         ">95% retention", "90-95% retention", "<90% retention",
         "92%", "Amber"),

    Risk("R23", "AI Governance and Regulatory Ethics Risk",
         "Regulatory", "Negative", 5, 9, 45, "HIGH",
         prob(5), 20.0, 80.0, 150.0, -1,
         "Chief Compliance Officer / AI Governance Team",
         "AI Governance Maturity Score",
         ">80% maturity", "60-80% maturity", "<60% maturity",
         "65%", "Amber"),

    # ── NEGATIVE MEDIUM ───────────────────────────────────────────────────────
    Risk("R15", "Legal Liability from Compliance Failures",
         "Regulatory", "Negative", 7, 6, 42, "MEDIUM",
         prob(7), 10.0, 40.0, 80.0, -1,
         "Chief Legal Officer",
         "Regulatory Compliance Audit Score",
         ">90% audit pass rate", "75-90% pass rate", "<75% pass rate",
         "82%", "Amber"),

    Risk("R5",  "Acquisition Overvaluation Risk",
         "Financial", "Negative", 5, 8, 40, "MEDIUM",
         prob(5), 50.0, 150.0, 250.0, -1,
         "Chief Financial Officer",
         "Revenue Synergy Achievement Rate",
         ">80% of targets met", "60-80% of targets met", "<60% of targets met",
         "N/A - integration pending", "Amber"),

    Risk("R6",  "Integration Overload from Multi-Acquisition Complexity",
         "Operational", "Negative", 5, 6, 30, "MEDIUM",
         prob(5), 5.0, 20.0, 40.0, -1,
         "Chief Technology Officer",
         "Integration Milestone Completion Rate",
         ">85% on schedule", "70-85% on schedule", "<70% on schedule",
         "72%", "Amber"),

    Risk("R7",  "Technology Architecture Incompatibility",
         "Technical", "Negative", 5, 6, 30, "MEDIUM",
         prob(5), 5.0, 20.0, 40.0, -1,
         "Chief Technology Officer",
         "Data Synchronization Error Rate",
         "<2% error rate", "2-5% error rate", ">5% error rate",
         "3.1%", "Amber"),

    Risk("R8",  "Cybersecurity Talent Shortage",
         "Operational", "Negative", 5, 6, 30, "MEDIUM",
         prob(5), 5.0, 18.0, 35.0, -1,
         "Chief Human Resources Officer",
         "Open Security Role Fill Rate",
         ">80% filled within 60 days", "60-80% filled", "<60% filled",
         "68%", "Amber"),

    Risk("R9",  "Post-Acquisition Organisational Disruption",
         "Operational", "Negative", 5, 6, 30, "MEDIUM",
         prob(5), 5.0, 18.0, 35.0, -1,
         "Chief Human Resources Officer",
         "Employee Retention Rate (Armis staff)",
         ">90% retention at 12 months", "80-90% retention", "<80% retention",
         "N/A - pre-close", "Amber"),

    Risk("R10", "Asset Visibility Data Accuracy Gaps",
         "Technical", "Negative", 5, 6, 30, "MEDIUM",
         prob(5), 3.0, 12.0, 25.0, -1,
         "Chief Product Officer",
         "Asset Discovery Accuracy Rate",
         ">98% accuracy", "95-98% accuracy", "<95% accuracy",
         "96.2%", "Amber"),

    Risk("R11", "Infrastructure Instability During Integration",
         "Operational", "Negative", 5, 6, 30, "MEDIUM",
         prob(5), 3.0, 12.0, 25.0, -1,
         "Chief Technology Officer",
         "Platform Uptime During Integration",
         ">99.5% uptime", "98-99.5% uptime", "<98% uptime",
         "99.1%", "Amber"),

    Risk("R12", "Security Vulnerabilities During Merger Integration",
         "Technical", "Negative", 5, 6, 30, "MEDIUM",
         prob(5), 5.0, 20.0, 40.0, -1,
         "Chief Information Security Officer",
         "Security Posture Score During Integration",
         ">85 security score", "70-85 security score", "<70 security score",
         "76", "Amber"),

    Risk("R14", "Global Regulatory Compliance Complexity",
         "Regulatory", "Negative", 5, 6, 30, "MEDIUM",
         prob(5), 3.0, 12.0, 25.0, -1,
         "Chief Compliance Officer",
         "Multi-Jurisdiction Compliance Coverage Rate",
         ">95% jurisdictions covered", "85-95% covered", "<85% covered",
         "88%", "Amber"),

    Risk("R16", "Infrastructure and Cloud Cost Expansion",
         "Financial", "Negative", 5, 6, 30, "MEDIUM",
         prob(5), 5.0, 20.0, 40.0, -1,
         "Chief Financial Officer",
         "Cloud Cost as % of Revenue",
         "<8% of revenue", "8-12% of revenue", ">12% of revenue",
         "9.4%", "Amber"),

    Risk("R24", "Third-Party Cloud Infrastructure Dependency",
         "Technical", "Negative", 5, 6, 30, "MEDIUM",
         prob(5), 3.0, 12.0, 25.0, -1,
         "Chief Technology Officer / Cloud Infrastructure Team",
         "Multi-Cloud Redundancy Coverage",
         ">80% workloads multi-cloud", "50-80% multi-cloud", "<50% multi-cloud",
         "42%", "Red"),

    Risk("R18", "Economic Slowdown Reducing Cybersecurity Spend",
         "Market", "Negative", 5, 6, 30, "MEDIUM",
         prob(5), 5.0, 20.0, 40.0, -1,
         "Chief Revenue Officer",
         "Pipeline Conversion Rate",
         ">25% conversion", "15-25% conversion", "<15% conversion",
         "21%", "Amber"),

    Risk("R4",  "Cybersecurity Market Consolidation Pressure",
         "Strategic", "Negative", 5, 6, 30, "MEDIUM",
         prob(5), 5.0, 18.0, 35.0, -1,
         "Chief Strategy Officer",
         "Market Share in OT/IoT Security",
         ">12% market share", "8-12% market share", "<8% market share",
         "9.1%", "Amber"),

    # ── NEGATIVE LOW ──────────────────────────────────────────────────────────
    Risk("R21", "Minor Operational Workflow Misalignment",
         "Operational", "Negative", 3, 4, 12, "LOW",
         prob(3), 0.5, 2.0, 5.0, -1,
         "Chief Operating Officer",
         "Cross-Team Process Alignment Score",
         ">85% aligned", "70-85% aligned", "<70% aligned",
         "74%", "Amber"),

    Risk("R22", "Short-Term Customer Transition Uncertainty",
         "Market", "Negative", 3, 2, 6, "LOW",
         prob(3), 0.5, 1.5, 4.0, -1,
         "Chief Revenue Officer",
         "Customer Satisfaction Score During Transition",
         ">4.0/5.0 CSAT", "3.5-4.0 CSAT", "<3.5 CSAT",
         "N/A - pre-close", "Amber"),

    Risk("R25", "Vendor Lock-In Limiting Technology Flexibility",
         "Technical", "Negative", 3, 6, 18, "LOW",
         prob(3), 1.0, 5.0, 12.0, -1,
         "Chief Information Officer / Enterprise Architecture",
         "Open Architecture Adoption Rate",
         ">70% open standards", "50-70% open standards", "<50% open standards",
         "55%", "Amber"),
]

# ─── STEP 1: ANALYTICAL EXPECTED VALUES ───────────────────────────────────────
def triangular_variance(lo: float, mo: float, hi: float) -> float:
    return (lo**2 + mo**2 + hi**2 - lo*mo - lo*hi - mo*hi) / 18.0

def analytical_metrics(risks: List[Risk]) -> pd.DataFrame:
    rows = []
    for r in risks:
        tri_mean = (r.low_m + r.mode_m + r.high_m) / 3.0
        tri_var  = triangular_variance(r.low_m, r.mode_m, r.high_m)
        ev       = r.sign * r.probability * tri_mean
        second   = r.probability * (tri_var + tri_mean**2)
        var      = second - (r.probability * tri_mean)**2
        rows.append({
            "risk_id":              r.risk_id,
            "risk_name":            r.name,
            "category":             r.category,
            "risk_type":            r.risk_type,
            "likelihood":           r.likelihood,
            "impact":               r.impact,
            "risk_score":           r.risk_score,
            "priority":             r.priority,
            "probability":          r.probability,
            "tri_mean_if_occurs_m": tri_mean,
            "expected_signed_m":    ev,
            "portfolio_variance":   var,
            "kri_status":           r.kri_status,
        })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_PROC / "analytical_metrics.csv", index=False)
    return df

# ─── STEP 2: MONTE CARLO SIMULATION ───────────────────────────────────────────
def run_monte_carlo(risks: List[Risk]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(SEED)
    sim: Dict[str, np.ndarray] = {}
    for r in risks:
        occ     = rng.binomial(1, r.probability, N_ITERATIONS)
        impacts = rng.triangular(r.low_m, r.mode_m, r.high_m, N_ITERATIONS)
        sim[r.risk_id] = r.sign * occ * impacts

    net = np.sum(np.column_stack(list(sim.values())), axis=1)
    # NOTE: Risks are modeled as independent processes. This simplifies computation
    # but likely understates true portfolio variance, as strategic risks (R1, R2, R13,
    # R17) are positively correlated during industry consolidation periods.
    # Future enhancement: introduce a copula-based correlation structure for
    # correlated strategic risks to capture joint tail risk more accurately.
    return sim, net

def summarize_mc(sim: Dict[str, np.ndarray], net: np.ndarray,
                 risks: List[Risk]) -> pd.DataFrame:
    rows = []
    for r in risks:
        arr = sim[r.risk_id]
        rows.append({
            "risk_id":   r.risk_id,
            "risk_name": r.name,
            "mean_m":    float(np.mean(arr)),
            "std_m":     float(np.std(arr, ddof=1)),
            "p05_m":     float(np.percentile(arr, 5)),
            "p95_m":     float(np.percentile(arr, 95)),
            "p_nonzero": float(np.mean(arr != 0)),
        })
    summary = pd.DataFrame(rows)

    # Portfolio row
    port = pd.DataFrame([{
        "risk_id":   "PORTFOLIO",
        "risk_name": "Net Portfolio Impact",
        "mean_m":    float(np.mean(net)),
        "std_m":     float(np.std(net, ddof=1)),
        "p05_m":     float(np.percentile(net, 5)),
        "p95_m":     float(np.percentile(net, 95)),
        "p_nonzero": float(np.mean(net != 0)),
    }])
    summary = pd.concat([summary, port], ignore_index=True)
    summary.to_csv(DATA_PROC / "monte_carlo_summary.csv", index=False)
    return summary

# ─── STEP 3: CATEGORY-LEVEL PORTFOLIO ANALYSIS ────────────────────────────────
def category_portfolios(sim: Dict[str, np.ndarray],
                        risks: List[Risk]) -> pd.DataFrame:
    cat_map: Dict[str, List[str]] = {}
    for r in risks:
        cat_map.setdefault(r.category, []).append(r.risk_id)

    rows = []
    for cat, ids in cat_map.items():
        arr = np.sum(np.column_stack([sim[i] for i in ids]), axis=1)
        rows.append({
            "category":  cat,
            "risk_ids":  ", ".join(ids),
            "mean_m":    float(np.mean(arr)),
            "std_m":     float(np.std(arr, ddof=1)),
            "p05_m":     float(np.percentile(arr, 5)),
            "p95_m":     float(np.percentile(arr, 95)),
            "p_loss":    float(np.mean(arr < 0)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_PROC / "category_portfolios.csv", index=False)
    return df

# ─── STEP 4: SYNTHETIC DATASET FOR ML ─────────────────────────────────────────
def build_synthetic_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n   = N_SYNTHETIC

    # 15 features aligned to KRI framework
    win_rate            = rng.uniform(0.15, 0.55, n)   # R1, R17
    ai_parity           = rng.uniform(0.55, 0.95, n)   # R2, R13
    reg_pipeline_growth = rng.uniform(-0.05, 0.50, n)  # R3
    integration_prog    = rng.uniform(0.10, 0.90, n)   # R6, R7
    talent_avail        = rng.uniform(0.40, 0.90, n)   # R8
    org_stability       = rng.uniform(0.50, 0.95, n)   # R9
    data_quality        = rng.uniform(0.85, 0.99, n)   # R10
    platform_uptime     = rng.uniform(0.96, 0.999, n)  # R11
    security_score      = rng.uniform(55.0, 95.0, n)   # R12
    compliance_complex  = rng.uniform(0.20, 0.90, n)   # R14, R15
    cloud_cost_var      = rng.uniform(0.05, 0.20, n)   # R16, R24
    customer_retention  = rng.uniform(0.82, 0.99, n)   # R17, R22
    ai_gov_maturity     = rng.uniform(0.40, 0.90, n)   # R23
    ot_market_growth    = rng.uniform(0.05, 0.35, n)   # R19
    cross_sell_conv     = rng.uniform(0.05, 0.30, n)   # R20

    # Latent escalation score encoding KRI logic
    latent = (
        (0.40 - win_rate)           * 7.0
      + (0.82 - ai_parity)          * 6.5
      + (0.12 - reg_pipeline_growth)* 1.5
      + (0.60 - integration_prog)   * 2.5
      + (0.70 - talent_avail)       * 1.8
      + (0.75 - org_stability)      * 1.5
      + (0.97 - data_quality)       * 3.0
      + (0.99 - platform_uptime)    * 4.0
      + (75.0 - security_score)     * 0.05
      + compliance_complex          * 1.2
      + (cloud_cost_var - 0.10)     * 2.0
      + (0.92 - customer_retention) * 3.5
      + (0.70 - ai_gov_maturity)    * 1.8
      + (0.15 - ot_market_growth)  * -1.0
      + (0.18 - cross_sell_conv)   * -1.0
      + rng.normal(0, 1.05, n)
    )

    escalation = (latent > 1.60).astype(int)

    df = pd.DataFrame({
        "win_rate":            win_rate,
        "ai_parity":           ai_parity,
        "reg_pipeline_growth": reg_pipeline_growth,
        "integration_prog":    integration_prog,
        "talent_avail":        talent_avail,
        "org_stability":       org_stability,
        "data_quality":        data_quality,
        "platform_uptime":     platform_uptime,
        "security_score":      security_score,
        "compliance_complex":  compliance_complex,
        "cloud_cost_var":      cloud_cost_var,
        "customer_retention":  customer_retention,
        "ai_gov_maturity":     ai_gov_maturity,
        "ot_market_growth":    ot_market_growth,
        "cross_sell_conv":     cross_sell_conv,
        "escalation":          escalation,
    })
    df.to_csv(DATA_PROC / "ml_training_data.csv", index=False)
    return df

# ─── STEP 5: ML MODEL ─────────────────────────────────────────────────────────
FEATURE_COLS = [
    "win_rate", "ai_parity", "reg_pipeline_growth", "integration_prog",
    "talent_avail", "org_stability", "data_quality", "platform_uptime",
    "security_score", "compliance_complex", "cloud_cost_var",
    "customer_retention", "ai_gov_maturity", "ot_market_growth",
    "cross_sell_conv",
]

def train_model(df: pd.DataFrame) -> dict:
    X = df[FEATURE_COLS]
    y = df["escalation"]

    model = RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=8,
        random_state=SEED, class_weight="balanced",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    acc   = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    auc   = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    pred  = cross_val_predict(model, X, y, cv=cv, method="predict")
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    model.fit(X, y)

    perm = permutation_importance(model, X, y, n_repeats=10, random_state=SEED)
    fi_df = pd.DataFrame({
        "feature":         FEATURE_COLS,
        "importance_mean": perm.importances_mean,
        "importance_std":  perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    fi_df.to_csv(DATA_PROC / "feature_importance.csv", index=False)

    return {
        "model":        model,
        "acc_mean":     float(np.mean(acc)),
        "acc_std":      float(np.std(acc, ddof=1)),
        "auc_mean":     float(np.mean(auc)),
        "auc_std":      float(np.std(auc, ddof=1)),
        "clf_report":   classification_report(y, pred),
        "conf_matrix":  confusion_matrix(y, pred),
        "full_auc":     float(roc_auc_score(y, proba)),
        "fi_df":        fi_df,
    }

def score_current_state(model: RandomForestClassifier) -> pd.DataFrame:
    """Current KRI values mapped to model features."""
    current = pd.DataFrame([{
        "win_rate":            0.31,   # Amber - R1
        "ai_parity":           0.78,   # Amber - R2
        "reg_pipeline_growth": 0.32,   # Green - R3
        "integration_prog":    0.48,   # Amber - R6/R7
        "talent_avail":        0.68,   # Amber - R8
        "org_stability":       0.80,   # Amber - R9
        "data_quality":        0.962,  # Amber - R10
        "platform_uptime":     0.991,  # Amber - R11
        "security_score":      76.0,   # Amber - R12
        "compliance_complex":  0.62,   # Amber - R14/R15
        "cloud_cost_var":      0.094,  # Amber - R16
        "customer_retention":  0.92,   # Amber - R17
        "ai_gov_maturity":     0.65,   # Amber - R23
        "ot_market_growth":    0.08,   # Amber - R19
        "cross_sell_conv":     0.12,   # Amber - R20
    }])
    prob_esc = float(model.predict_proba(current[FEATURE_COLS])[0, 1])
    current["escalation_probability"] = prob_esc
    current.to_csv(DATA_PROC / "current_portfolio_state.csv", index=False)
    return current, prob_esc

# ─── STEP 6: PLOTS ────────────────────────────────────────────────────────────
COLORS = {"HIGH": "#E74C3C", "MEDIUM": "#F39C12", "LOW": "#27AE60",
          "Positive": "#2ECC71", "Negative": "#E74C3C"}

def plot_risk_heatmap(risks: List[Risk]) -> None:
    """Likelihood-Impact scatter heatmap with all 25 risks."""
    fig, ax = plt.subplots(figsize=(14, 9))

    # Background zones
    ax.fill_between([0, 3],  [0, 0], [10, 10], color="#27AE60", alpha=0.08)
    ax.fill_between([3, 6],  [0, 0], [10, 10], color="#F39C12", alpha=0.08)
    ax.fill_between([6, 10], [0, 0], [10, 10], color="#E74C3C", alpha=0.08)

    # Scatter
    color_map = {"HIGH": "#E74C3C", "MEDIUM": "#F39C12", "LOW": "#27AE60"}
    marker_map = {"Positive": "^", "Negative": "o"}

    from collections import defaultdict
    jitter_tracker: dict = defaultdict(int)

    for r in risks:
        key  = (r.likelihood, r.impact)
        cnt  = jitter_tracker[key]
        jitter_tracker[key] += 1
        angle  = cnt * (2 * np.pi / 6)
        radius = 0.22 * cnt
        xp = r.likelihood + radius * np.cos(angle)
        yp = r.impact     + radius * np.sin(angle)
        ax.scatter(xp, yp,
                   color=color_map[r.priority],
                   marker=marker_map[r.risk_type],
                   s=180, zorder=5,
                   edgecolors="black", linewidths=0.6)
        ax.annotate(r.risk_id, (xp, yp),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, fontweight="bold")

    ax.set_xlim(0.5, 10)
    ax.set_ylim(0.5, 10)
    ax.set_xticks([1, 3, 5, 7, 9])
    ax.set_xticklabels(["1\nVery\nUnlikely", "3\nSomewhat\nUnlikely",
                         "5\n50-50", "7\nSomewhat\nLikely", "9\nVery\nLikely"], fontsize=9)
    ax.set_yticks([1, 2, 4, 6, 8, 9])
    ax.set_yticklabels(["1\nVery Low", "2\nSomewhat Low", "4\nModerate",
                         "6\nSomewhat High", "8\nHigh", "9\nExtremely\nHigh"], fontsize=9)
    ax.set_xlabel("Likelihood Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Impact Score", fontsize=12, fontweight="bold")
    ax.set_title("Figure 1: Enterprise Risk Heatmap - ServiceNow-Armis Acquisition (25 Risks)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3, linestyle="--")

    legend_els = [
        mpatches.Patch(color="#E74C3C", label="High Priority (score ≥45)"),
        mpatches.Patch(color="#F39C12", label="Medium Priority (score 20-44)"),
        mpatches.Patch(color="#27AE60", label="Low Priority (score <20)"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
                   markersize=9, label="Positive Risk"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=9, label="Negative Risk"),
    ]
    ax.legend(handles=legend_els, loc="upper left", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(REPORT / "figure1_risk_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: figure1_risk_heatmap.png")


def plot_mc_net_distribution(net: np.ndarray) -> None:
    """Monte Carlo net portfolio impact distribution."""
    mean_v = float(np.mean(net))
    p05    = float(np.percentile(net, 5))
    p95    = float(np.percentile(net, 95))
    p_loss = float(np.mean(net < 0))

    fig, ax = plt.subplots(figsize=(12, 6))
    n_neg = net[net < 0]
    n_pos = net[net >= 0]
    ax.hist(n_neg, bins=50, color="#E74C3C", alpha=0.65, edgecolor="black",
            linewidth=0.3, label="Net Loss scenarios")
    ax.hist(n_pos, bins=50, color="#27AE60", alpha=0.65, edgecolor="black",
            linewidth=0.3, label="Net Gain scenarios")
    ax.axvline(0,      color="black",   lw=2,     linestyle="-",  label="Break-even ($0)")
    ax.axvline(mean_v, color="#2C3E50", lw=2,     linestyle="--",
               label=f"Mean: ${mean_v:+,.1f}M")
    ax.axvline(p05,    color="#E67E22", lw=1.8,   linestyle=":",
               label=f"5th pct: ${p05:+,.1f}M")
    ax.axvline(p95,    color="#1ABC9C", lw=1.8,   linestyle=":",
               label=f"95th pct: ${p95:+,.1f}M")

    ylim = ax.get_ylim()[1]
    ax.text(p05 * 1.05, ylim * 0.88,
            f"P(Net Loss)\n{p_loss:.1%}", color="#C0392B",
            fontsize=11, fontweight="bold", ha="center")
    ax.text(p95 * 0.85, ylim * 0.88,
            f"P(Net Gain)\n{1-p_loss:.1%}", color="#1E8449",
            fontsize=11, fontweight="bold", ha="center")

    ax.set_xlabel("Net Annual Portfolio Impact (USD Millions)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency (out of 10,000 iterations)", fontsize=12)
    ax.set_title(
        f"Figure 2: Monte Carlo Net Portfolio Impact Distribution\n"
        f"(n=10,000 iterations, seed=42, 25 risks)",
        fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(REPORT / "figure2_mc_net_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: figure2_mc_net_distribution.png")


def plot_individual_risk_distributions(sim: Dict[str, np.ndarray],
                                        risks: List[Risk]) -> None:
    """Box plots for all 25 risks."""
    ids    = [r.risk_id for r in risks]
    data   = [sim[i] for i in ids]
    colors = [COLORS[r.priority] for r in risks]

    fig, ax = plt.subplots(figsize=(18, 7))
    bp = ax.boxplot(data, labels=ids, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    ax.axhline(0, color="black", lw=1.2, linestyle="--", alpha=0.6)
    ax.set_xlabel("Risk ID", fontsize=12, fontweight="bold")
    ax.set_ylabel("Simulated Annual Impact (USD Millions)", fontsize=12)
    ax.set_title(
        "Figure 3: Individual Risk Impact Distributions (25 Risks, Monte Carlo)",
        fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, linestyle="--")

    legend_els = [
        mpatches.Patch(color="#E74C3C", alpha=0.75, label="High Priority"),
        mpatches.Patch(color="#F39C12", alpha=0.75, label="Medium Priority"),
        mpatches.Patch(color="#27AE60", alpha=0.75, label="Low Priority"),
    ]
    ax.legend(handles=legend_els, fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORT / "figure3_individual_distributions.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: figure3_individual_distributions.png")


def plot_category_portfolios(cat_df: pd.DataFrame) -> None:
    """Category-level portfolio expected values."""
    df = cat_df.sort_values("mean_m")
    colors = ["#E74C3C" if v < 0 else "#27AE60" for v in df["mean_m"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(df["category"], df["mean_m"], color=colors,
                   edgecolor="black", linewidth=0.5, alpha=0.82)
    ax.axvline(0, color="black", lw=1.5)
    for bar, v in zip(bars, df["mean_m"]):
        xpos = v + (3 if v >= 0 else -3)
        ha   = "left" if v >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"${v:+,.1f}M", va="center", ha=ha,
                fontsize=10, fontweight="bold")

    ax.set_xlabel("Expected Annual Impact (USD Millions)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Figure 4: Expected Impact by Risk Category Portfolio",
        fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", axis="x")
    plt.tight_layout()
    plt.savefig(REPORT / "figure4_category_portfolios.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: figure4_category_portfolios.png")


def plot_feature_importance(fi_df: pd.DataFrame) -> None:
    """Permutation feature importance with KRI mapping."""
    ordered = fi_df.sort_values("importance_mean")
    colors  = ["#3498DB"] * len(ordered)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(ordered["feature"], ordered["importance_mean"],
            xerr=ordered["importance_std"], color=colors,
            edgecolor="black", linewidth=0.4, alpha=0.85,
            error_kw={"elinewidth": 1.2, "capsize": 3})
    ax.set_xlabel("Mean Permutation Importance", fontsize=12, fontweight="bold")
    ax.set_title(
        "Figure 5: ML Model - Permutation Feature Importance\n"
        "(Random Forest, 15 KRI-Aligned Features)",
        fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", axis="x")
    plt.tight_layout()
    plt.savefig(REPORT / "figure5_feature_importance.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: figure5_feature_importance.png")


def plot_kri_dashboard(risks: List[Risk]) -> None:
    """KRI status traffic-light dashboard for all 25 risks."""
    status_color = {"Green": "#27AE60", "Amber": "#F39C12", "Red": "#E74C3C"}
    ids      = [r.risk_id for r in risks]
    statuses = [r.kri_status for r in risks]
    colors   = [status_color[s] for s in statuses]

    fig, ax = plt.subplots(figsize=(16, 5))
    for i, (rid, col) in enumerate(zip(ids, colors)):
        circle = plt.Circle((i, 0), 0.38, color=col, zorder=5)
        ax.add_patch(circle)
        ax.text(i, 0, rid, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=6)

    ax.set_xlim(-0.8, len(ids) - 0.2)
    ax.set_ylim(-0.8, 0.8)
    ax.axis("off")
    ax.set_title(
        "Figure 6: KRI Status Dashboard - All 25 Risks (Current State)",
        fontsize=13, fontweight="bold")

    legend_els = [
        mpatches.Patch(color="#27AE60", label="Green: Within threshold"),
        mpatches.Patch(color="#F39C12", label="Amber: Warning zone"),
        mpatches.Patch(color="#E74C3C", label="Red: Critical - action required"),
    ]
    ax.legend(handles=legend_els, loc="lower center",
              ncol=3, fontsize=10, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.35))
    plt.tight_layout()
    plt.savefig(REPORT / "figure6_kri_dashboard.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: figure6_kri_dashboard.png")


def plot_confusion_matrix(conf_mat: np.ndarray, ml_results: dict) -> None:
    """Confusion matrix for ML model."""
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat,
        display_labels=["No Escalation", "Escalation"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"Figure 7: ML Model Confusion Matrix\n"
        f"(CV Accuracy: {ml_results['acc_mean']:.1%}, "
        f"ROC-AUC: {ml_results['auc_mean']:.1%})",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORT / "figure7_confusion_matrix.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: figure7_confusion_matrix.png")


def plot_scenario_analysis(risks: List[Risk]) -> None:
    """Scenario analysis for high-priority risks."""
    high = [r for r in risks if r.priority == "HIGH"]
    scenarios = ["Best Case", "Most Likely", "Worst Case"]
    s_colors  = ["#27AE60", "#F39C12", "#E74C3C"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for idx, r in enumerate(high[:9]):
        ax = axes[idx // 3][idx % 3]
        vals = [r.sign * r.low_m,
                r.sign * r.mode_m,
                r.sign * r.high_m]
        bars = ax.bar(scenarios, [abs(v) for v in vals],
                      color=s_colors, edgecolor="black",
                      linewidth=0.5, alpha=0.82)
        for bar, v in zip(bars, vals):
            sign = "+" if v >= 0 else "-"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{sign}${abs(v):.0f}M",
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
        ax.set_title(f"{r.risk_id}: {r.name[:30]}",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel("Impact ($M)", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.suptitle(
        "Figure 8: Scenario Analysis - High-Priority Risks\n"
        "(Best Case / Most Likely / Worst Case)",
        fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(REPORT / "figure8_scenario_analysis.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: figure8_scenario_analysis.png")


# ─── STEP 7: EXECUTIVE SUMMARY ────────────────────────────────────────────────
def write_executive_summary(anal_df, mc_df, cat_df, ml_results, prob_esc):
    net_row  = mc_df[mc_df["risk_id"] == "PORTFOLIO"].iloc[0]
    top3_fi  = ml_results["fi_df"]["feature"].head(3).tolist()

    lines = [
        "EXECUTIVE SUMMARY - SERVICENOW/ARMIS ENTERPRISE RISK ANALYTICS",
        "=" * 70,
        "",
        "QUANTITATIVE RISK REGISTER (25 RISKS)",
        f"  Total risks: 25 (3 positive, 22 negative)",
        f"  Priority distribution: 8 HIGH, 14 MEDIUM, 3 LOW",
        "",
        "ANALYTICAL EXPECTED VALUES",
        f"  Sum positive expected impacts: "
        f"${anal_df[anal_df['risk_type']=='Positive']['expected_signed_m'].sum():+.1f}M",
        f"  Sum negative expected impacts: "
        f"${anal_df[anal_df['risk_type']=='Negative']['expected_signed_m'].sum():+.1f}M",
        f"  Net analytical expected impact: "
        f"${anal_df['expected_signed_m'].sum():+.1f}M",
        "",
        "MONTE CARLO SIMULATION (n=10,000, seed=42)",
        f"  Net expected annual impact:   ${net_row['mean_m']:+,.1f}M",
        f"  Standard deviation:           ${net_row['std_m']:,.1f}M",
        f"  5th percentile (downside):    ${net_row['p05_m']:+,.1f}M",
        f"  95th percentile (upside):     ${net_row['p95_m']:+,.1f}M",
        f"  90% confidence interval:      ${net_row['p05_m']:+,.1f}M to ${net_row['p95_m']:+,.1f}M",
        f"  Probability of net loss:      {p_loss:.1%}",
        f"  Probability of net gain:      {p_gain:.1%}",
        "",
        "CATEGORY PORTFOLIO BREAKDOWN",
    ]

    for _, row in cat_df.sort_values("mean_m").iterrows():
        lines.append(f"  {row['category']:<20s}: ${row['mean_m']:+,.1f}M "
                     f"(P(loss): {row['p_loss']:.1%})")

    lines += [
        "",
        "MACHINE LEARNING MODEL",
        f"  CV Accuracy:        {ml_results['acc_mean']:.1%} (SD: {ml_results['acc_std']:.1%})",
        f"  CV ROC-AUC:         {ml_results['auc_mean']:.1%} (SD: {ml_results['auc_std']:.1%})",
        f"  Escalation prob (current state): {prob_esc:.1%}",
        f"  Top 3 features:     {', '.join(top3_fi)}",
        "",
        "KRI CURRENT STATUS",
        "  Green: 1 risk (R3 - Regulatory Pipeline Growth)",
        "  Red:   1 risk (R24 - Multi-Cloud Redundancy)",
        "  Amber: 23 risks",
        "",
        "CLASSIFICATION REPORT",
        ml_results["clf_report"],
    ]

    text = "\n".join(lines)
    (DATA_PROC / "executive_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    return text


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("STEP 1: Analytical expected values")
    print("="*70)
    anal_df = analytical_metrics(RISKS)
    print(anal_df[["risk_id","risk_name","expected_signed_m","portfolio_variance"]].to_string())
    print(f"\nNet analytical expected impact: ${anal_df['expected_signed_m'].sum():+.2f}M")

    print("\n" + "="*70)
    print("STEP 2: Monte Carlo simulation")
    print("="*70)
    sim, net = run_monte_carlo(RISKS)
    mc_df    = summarize_mc(sim, net, RISKS)

    # Lock in the key numbers
    net_row  = mc_df[mc_df["risk_id"] == "PORTFOLIO"].iloc[0]
    p_loss   = float(np.mean(net < 0))
    p_gain   = float(np.mean(net >= 0))
    print(f"\nNET PORTFOLIO (LOCKED NUMBERS):")
    print(f"  Mean:          ${net_row['mean_m']:+,.2f}M")
    print(f"  Std Dev:       ${net_row['std_m']:,.2f}M")
    print(f"  5th pct:       ${net_row['p05_m']:+,.2f}M")
    print(f"  95th pct:      ${net_row['p95_m']:+,.2f}M")
    print(f"  P(net loss):   {p_loss:.4f}  ({p_loss:.1%})")
    print(f"  P(net gain):   {p_gain:.4f}  ({p_gain:.1%})")

    # Save p_loss separately for report use
    pd.DataFrame([{
        "p_loss": p_loss, "p_gain": p_gain,
        "net_mean": net_row["mean_m"],
        "net_std":  net_row["std_m"],
        "net_p05":  net_row["p05_m"],
        "net_p95":  net_row["p95_m"],
    }]).to_csv(DATA_PROC / "portfolio_key_stats.csv", index=False)

    print("\n" + "="*70)
    print("STEP 3: Category portfolios")
    print("="*70)
    cat_df = category_portfolios(sim, RISKS)
    print(cat_df.to_string())

    print("\n" + "="*70)
    print("STEP 4: Synthetic dataset")
    print("="*70)
    syn_df = build_synthetic_dataset()
    print(f"  {len(syn_df)} obs | escalation rate: {syn_df['escalation'].mean():.1%}")

    print("\n" + "="*70)
    print("STEP 5: ML model")
    print("="*70)
    ml_results = train_model(syn_df)
    print(f"  Accuracy: {ml_results['acc_mean']:.4f} ± {ml_results['acc_std']:.4f}")
    print(f"  ROC-AUC:  {ml_results['auc_mean']:.4f} ± {ml_results['auc_std']:.4f}")
    print("\nTop 5 features:")
    print(ml_results["fi_df"].head(5).to_string())

    print("\n" + "="*70)
    print("STEP 6: Current portfolio scoring")
    print("="*70)
    current_df, prob_esc = score_current_state(ml_results["model"])
    print(f"  Escalation probability: {prob_esc:.4f} ({prob_esc:.1%})")

    print("\n" + "="*70)
    print("STEP 7: Generating all figures")
    print("="*70)
    plot_risk_heatmap(RISKS)
    plot_mc_net_distribution(net)
    plot_individual_risk_distributions(sim, RISKS)
    plot_category_portfolios(cat_df)
    plot_feature_importance(ml_results["fi_df"])
    plot_kri_dashboard(RISKS)
    plot_confusion_matrix(ml_results["conf_matrix"], ml_results)
    plot_scenario_analysis(RISKS)

    print("\n" + "="*70)
    print("STEP 8: Executive summary")
    print("="*70)
    write_executive_summary(anal_df, mc_df, cat_df, ml_results, prob_esc)

    print("\n" + "="*70)
    print("ALL OUTPUTS SAVED")
    print("="*70)
    for f in sorted((DATA_PROC).iterdir()):
        print(f"  data/processed/{f.name}")
    for f in sorted(REPORT.iterdir()):
        print(f"  report/{f.name}")


if __name__ == "__main__":
    main()
