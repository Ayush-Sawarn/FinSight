"""
generate_report.py
------------------
Generates a professional Business Performance Report as HTML.
Embeds all charts and KPI metrics into a single self-contained file.
"""

import os
import base64
import pandas as pd
from datetime import datetime

os.makedirs("reports/output", exist_ok=True)


def img_to_base64(path: str) -> str:
    """Convert image file to base64 string for HTML embedding."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_html_report(
    metrics_df: pd.DataFrame,
    purpose_df: pd.DataFrame,
    forecast_results: dict,
    recommendations: list
) -> str:
    """Build full HTML report string."""

    today = datetime.today().strftime("%B %d, %Y")

    # ── KPI Summary Cards ──
    monthly = forecast_results.get("monthly_kpis", pd.DataFrame())
    if not monthly.empty:
        total_loans = f"{monthly['total_loans_disbursed'].sum():,.0f}"
        avg_default = f"{monthly['default_rate_%'].mean()*100:.2f}%"
        total_loans_count = f"{monthly['loan_count'].sum():,.0f}"
        forecast_loans = forecast_results.get("loans_disbursed_forecast")
        forecast_val = f"{forecast_loans.mean():,.0f}" if forecast_loans is not None else "N/A"
    else:
        total_loans = avg_default = total_loans_count = forecast_val = "N/A"

    # ── Metrics Table ──
    metrics_html = ""
    if metrics_df is not None and not metrics_df.empty:
        metrics_html = metrics_df.to_html(classes="table", border=0, float_format="{:.4f}".format)

    # ── Channel Table ──
    purpose_html = ""
    if purpose_df is not None and not purpose_df.empty:
        purpose_display = purpose_df.copy()
        purpose_display["Default_Rate_%"] = purpose_display["Default_Rate_%"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
        purpose_display["Avg_Loan_Amount"] = purpose_display["Avg_Loan_Amount"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        purpose_html = purpose_display.to_html(classes="table", border=0, index=False)

    # ── Recommendations HTML ──
    priority_colors = {"HIGH": "#b85c50", "MEDIUM": "#c28e3e", "INFO": "#577590"}
    recs_html = ""
    for rec in recommendations:
        color = priority_colors.get(rec["priority"], "#9E9E9E")
        recs_html += f"""
        <div class="rec-card" style="border-left: 5px solid {color};">
            <span class="badge" style="background:{color};">{rec['priority']}</span>
            <strong>{rec['area']}</strong>
            <p><em>Insight:</em> {rec['insight']}</p>
            <p><em>Action:</em> {rec['action']}</p>
        </div>"""

    # ── Embed Images ──
    def img_tag(filename: str, alt: str = "") -> str:
        b64 = img_to_base64(f"reports/figures/{filename}")
        if not b64:
            return f"<p style='color:#999;font-style:italic;'>Chart not yet generated: {filename}</p>"
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="width:100%;border-radius:8px;margin:10px 0;">'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FinSight — Business Performance Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #F5F7FA; color: #333; }}
  .header {{ background: linear-gradient(135deg, #1565C0, #0D47A1); color: white; padding: 40px; }}
  .header h1 {{ font-size: 2.2em; font-weight: 700; }}
  .header p {{ opacity: 0.85; margin-top: 6px; font-size: 1em; }}
  .container {{ max-width: 1100px; margin: 30px auto; padding: 0 20px; }}
  .section {{ background: white; border-radius: 12px; padding: 28px; margin-bottom: 28px;
              box-shadow: 0 2px 12px rgba(0,0,0,0.07); }}
  .section h2 {{ font-size: 1.3em; color: #1565C0; border-bottom: 2px solid #E3F2FD;
                 padding-bottom: 10px; margin-bottom: 18px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 10px; }}
  .kpi-card {{ background: linear-gradient(135deg, #E3F2FD, #BBDEFB); border-radius: 10px;
               padding: 20px; text-align: center; }}
  .kpi-card .value {{ font-size: 1.6em; font-weight: 700; color: #1565C0; }}
  .kpi-card .label {{ font-size: 0.82em; color: #555; margin-top: 5px; }}
  .table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  .table th {{ background: #1565C0; color: white; padding: 10px 14px; text-align: left; }}
  .table td {{ padding: 9px 14px; border-bottom: 1px solid #EEE; }}
  .table tr:hover {{ background: #F5F7FA; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .rec-card {{ background: #FAFAFA; border-radius: 8px; padding: 16px 20px;
               margin-bottom: 14px; }}
  .rec-card p {{ margin-top: 6px; font-size: 0.9em; color: #555; line-height: 1.5; }}
  .badge {{ display: inline-block; color: white; font-size: 0.75em; font-weight: 700;
             padding: 3px 10px; border-radius: 12px; margin-right: 8px; }}
  .footer {{ text-align: center; color: #999; font-size: 0.82em; padding: 20px 0 40px; }}
</style>
</head>
<body>

<div class="header">
  <h1>FinSight — Business Performance Report</h1>
  <p>Consumer Finance Analytics Dashboard &nbsp;|&nbsp; Generated: {today}</p>
</div>

<div class="container">

  <!-- KPI Summary -->
  <div class="section">
    <h2>Executive KPI Summary</h2>
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="value">{total_loans}</div>
        <div class="label">Total Funded Amount</div>
      </div>
      <div class="kpi-card">
        <div class="value">{avg_default}</div>
        <div class="label">Avg Monthly Default Rate</div>
      </div>
      <div class="kpi-card">
        <div class="value">{total_loans_count}</div>
        <div class="label">Loans in Dataset</div>
      </div>
      <div class="kpi-card">
        <div class="value">{forecast_val}</div>
        <div class="label">Avg Forecasted Monthly Loans</div>
      </div>
    </div>
  </div>

  <!-- Data Overview -->
  <div class="section">
    <h2>Data Overview — Loan Portfolio</h2>
    {img_tag("target_distribution.png", "Default Distribution")}
    <div class="chart-grid" style="margin-top:20px;">
      {img_tag("numeric_distributions.png", "Feature Distributions")}
      {img_tag("default_rate_by_age.png", "Default Rate by Age")}
    </div>
    {img_tag("correlation_heatmap.png", "Correlation Heatmap")}
  </div>

  <!-- Forecasting -->
  <div class="section">
    <h2>Business Forecasting — Next 6 Months</h2>
    {img_tag("forecast_loans_disbursed.png", "Loans Forecast")}
    <div class="chart-grid" style="margin-top:20px;">
      {img_tag("forecast_default_rate.png", "Default Rate Forecast")}
      {img_tag("forecast_new_customers.png", "Loan Count Forecast")}
    </div>
  </div>

  <!-- Anomaly Detection -->
  <div class="section">
    <h2>Anomaly Detection</h2>
    {img_tag("anomaly_detection.png", "Anomaly Detection")}
  </div>

  <!-- Model Performance -->
  <div class="section">
    <h2>Default Prediction Model Performance</h2>
    {metrics_html}
    <div class="chart-grid" style="margin-top:20px;">
      {img_tag("roc_curves.png", "ROC Curves")}
      {img_tag("feature_importance_random_forest.png", "Feature Importance")}
    </div>
  </div>

  <!-- Loan Purpose Analysis -->
  <div class="section">
    <h2>Loan Purpose Analysis</h2>
    {purpose_html}
    {img_tag("loan_purpose_performance.png", "Loan Purpose Performance")}
  </div>

  <!-- Recommendations -->
  <div class="section">
    <h2>Business Insights & Recommendations</h2>
    {recs_html}
  </div>

</div>

<div class="footer">
  FinSight Analytics Platform &nbsp;|&nbsp; Confidential &nbsp;|&nbsp; {today}
</div>
</body>
</html>"""

    return html


def save_report(html: str, filename: str = "reports/output/business_performance_report.html"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport saved: {filename}")


if __name__ == "__main__":
    # Standalone test with dummy data
    dummy_metrics = pd.DataFrame({
        "Accuracy": [0.85], "Precision": [0.82],
        "Recall": [0.78], "F1-Score": [0.80], "ROC-AUC": [0.89]
    }, index=["Random Forest"])
    dummy_purpose = pd.DataFrame({
        "Purpose": ["Personal", "Home"],
        "Resolved_Loans": [12000, 8000],
        "Total_Loans": [15000, 8000],
        "Default_Rate_%": [0.15, 0.22],
        "Avg_Loan_Amount": [9000, 12000],
    })
    html = build_html_report(dummy_metrics, dummy_purpose, {}, [])
    save_report(html)
