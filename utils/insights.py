"""
insights.py
-----------
Analyzes Lending Club loan-purpose segments, detects monthly anomalies,
and generates rule-based business recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

os.makedirs("reports/figures", exist_ok=True)

PALETTE = ["#246b68", "#b85c50", "#577590", "#c28e3e", "#786a9b"]


def analyze_loan_purpose(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize actual Lending Club loan volume and resolved outcomes by purpose."""
    if "purpose" not in df.columns or "loan_status" not in df.columns:
        raise ValueError("Lending Club data must contain purpose and loan_status columns.")
    amount_col = "funded_amnt" if "funded_amnt" in df.columns else "loan_amnt"
    if amount_col not in df.columns:
        raise ValueError("Lending Club data must include funded_amnt or loan_amnt.")

    work = df[["purpose", "loan_status", amount_col]].copy()
    status = work["loan_status"].astype(str).str.strip().str.lower()
    paid = status.str.contains("fully paid", regex=False)
    defaulted = status.str.contains("charged off", regex=False) | status.eq("default")
    work["resolved_default"] = np.select([defaulted, paid], [1.0, 0.0], default=np.nan)
    work["loan_amount"] = pd.to_numeric(work[amount_col], errors="coerce")
    work["Purpose"] = work["purpose"].fillna("Unknown").astype(str).str.replace("_", " ").str.title()

    result = work.groupby("Purpose", dropna=False).agg(
        Total_Loans=("loan_status", "size"),
        default_rate=("resolved_default", "mean"),
        Resolved_Loans=("resolved_default", "count"),
        Avg_Loan_Amount=("loan_amount", "mean"),
    ).rename(columns={"default_rate": "Default_Rate_%"}).reset_index().sort_values("Total_Loans", ascending=False)
    return result


def plot_loan_purpose_performance(purpose_df: pd.DataFrame):
    """Plot observed loan volume and resolved default rate by loan purpose."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(purpose_df["Purpose"], purpose_df["Total_Loans"], color=PALETTE[0])
    axes[0].invert_yaxis()
    axes[0].set_title("Loan Count by Purpose", fontweight="bold")
    axes[0].set_xlabel("Loans")
    axes[1].bar(purpose_df["Purpose"], purpose_df["Default_Rate_%"] * 100, color=PALETTE[1])
    axes[1].set_title("Resolved Default Rate by Purpose", fontweight="bold")
    axes[1].set_ylabel("Default Rate (%)")
    axes[1].tick_params(axis="x", rotation=35)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("reports/figures/loan_purpose_performance.png", bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# 1. Channel Performance Analysis
# ─────────────────────────────────────────────
def detect_anomalies(monthly_kpis: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score based anomaly detection on monthly KPIs.
    Flags months where any KPI deviates > 2 std deviations.
    """
    kpi_cols = ["total_loans_disbursed", "default_rate_%", "loan_count"]
    anomalies = pd.DataFrame(index=monthly_kpis.index)

    for col in kpi_cols:
        if col in monthly_kpis.columns:
            z_scores = (monthly_kpis[col] - monthly_kpis[col].mean()) / monthly_kpis[col].std()
            anomalies[f"{col}_zscore"] = z_scores.round(3)
            anomalies[f"{col}_anomaly"] = z_scores.abs() > 2.0

    flagged = anomalies[anomalies.filter(like="_anomaly").any(axis=1)]
    print(f"\nAnomaly detection: {len(flagged)} months flagged out of {len(monthly_kpis)}")
    return anomalies


def plot_anomaly_detection(monthly_kpis: pd.DataFrame, anomalies: pd.DataFrame):
    """Highlight anomalous months on the KPI time-series."""
    col = "total_loans_disbursed"
    if col not in monthly_kpis.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly_kpis.index, monthly_kpis[col],
            color=PALETTE[0], linewidth=2, label="Loan Volume")

    # Highlight anomalies
    anomaly_flag_col = f"{col}_anomaly"
    if anomaly_flag_col in anomalies.columns:
        anomaly_dates = anomalies[anomalies[anomaly_flag_col]].index
        ax.scatter(anomaly_dates, monthly_kpis.loc[anomaly_dates, col],
                   color=PALETTE[1], s=100, zorder=5, label="Anomaly Detected", marker="^")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title("Loan Volume — Anomaly Detection", fontsize=13, fontweight="bold")
    ax.set_ylabel("Funded Amount")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("reports/figures/anomaly_detection.png")
    plt.close()
    print("Saved: reports/figures/anomaly_detection.png")


# ─────────────────────────────────────────────
# 3. Business Recommendations Engine
# ─────────────────────────────────────────────
def generate_recommendations(
    purpose_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    forecast_results: dict
) -> list:
    """
    Rule-based engine that generates actionable business insights.
    Returns a list of recommendation strings.
    """
    recs = []

    # Rule 1: Flag purposes above the portfolio default rate. Ignore tiny groups.
    eligible = purpose_df[purpose_df["Resolved_Loans"] >= 100]
    portfolio_default_rate = (
        (eligible["Default_Rate_%"] * eligible["Resolved_Loans"]).sum()
        / eligible["Resolved_Loans"].sum()
        if not eligible.empty else np.nan
    )
    high_risk = eligible[eligible["Default_Rate_%"] > portfolio_default_rate]
    for _, row in high_risk.iterrows():
        recs.append({
            "priority": "HIGH",
            "area": "Default Risk",
            "insight": f"'{row['Purpose']}' loans have a resolved default rate of {row['Default_Rate_%']*100:.1f}%, above the portfolio rate of {portfolio_default_rate*100:.1f}%.",
            "action": f"Review underwriting and borrower characteristics for the {row['Purpose'].lower()} segment."
        })

    # Rule 3: Forecast-based alerts
    if "default_rate_forecast" in forecast_results:
        fc = forecast_results["default_rate_forecast"]
        if fc.mean() > 0.22:
            recs.append({
                "priority": "HIGH",
                "area": "Forecast Alert",
                "insight": f"Forecasted default rate for next 6 months averages {fc.mean()*100:.1f}% — above 22% warning level.",
                "action": "Initiate proactive risk mitigation: enhanced due diligence, early-warning outreach to at-risk borrowers."
            })

    # Rule 4: Model performance
    if metrics_df is not None and "ROC-AUC" in metrics_df.columns:
        best_auc = metrics_df["ROC-AUC"].max()
        recs.append({
            "priority": "INFO",
            "area": "Model Performance",
            "insight": f"Best prediction model achieves ROC-AUC of {best_auc:.3f}.",
            "action": "Deploy model to production scoring pipeline. Schedule quarterly retraining."
        })

    print(f"\nGenerated {len(recs)} business recommendations")
    return recs


def print_recommendations(recs: list):
    """Pretty-print recommendations to console."""
    print("\n" + "-" * 60)
    print("  BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("-" * 60)
    for i, rec in enumerate(recs, 1):
        print(f"\n[{rec['priority']}] {rec['area']}")
        print(f"   Insight : {rec['insight']}")
        print(f"   Action  : {rec['action']}")
    print("\n" + "-" * 60)


if __name__ == "__main__":
    from models.forecasting import load_lending_club_data

    purpose_df = analyze_loan_purpose(load_lending_club_data())
    plot_loan_purpose_performance(purpose_df)
    print_recommendations(generate_recommendations(purpose_df, None, {}))
