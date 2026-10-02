"""
main.py
-------
Master pipeline runner for FinSight.
Runs all modules in order and generates the final HTML report.

Usage:
    python main.py
"""

import sys
import os
sys.path.append(".")

print("=" * 60)
print("FinSight: Business Performance Analytics Pipeline")
print("=" * 60)


def main():
    from models.forecasting import load_lending_club_data, run_forecasting
    lending_club_df = load_lending_club_data()

    # ── Step 1: Load & Clean Data ──
    print("\n[1/5] Loading and preprocessing data...")
    from data.data_loader import get_processed_data, load_raw_data
    df_raw = load_raw_data()
    X_train, X_test, y_train, y_test = get_processed_data()

    # ── Step 2: EDA ──
    print("\n[2/5] Running Exploratory Data Analysis...")
    from utils.eda import run_full_eda
    run_full_eda(df_raw)

    # ── Step 3: Forecasting ──
    print("\n[3/5] Forecasting Lending Club portfolio KPIs...")
    forecast_results = run_forecasting(lending_club_df)

    # ── Step 4: Default Prediction Model ──
    print("\n[4/5] Training Default Prediction Models...")
    from models.default_predictor import run_default_prediction_pipeline
    model_results = run_default_prediction_pipeline(X_train, X_test, y_train, y_test)

    # ── Step 5: Insights + Report ──
    print("\n[5/5] Generating Business Insights and Report...")
    from utils.insights import (
        analyze_loan_purpose,
        plot_loan_purpose_performance,
        detect_anomalies,
        plot_anomaly_detection,
        generate_recommendations,
        print_recommendations
    )
    from reports.generate_report import build_html_report, save_report

    purpose_df = analyze_loan_purpose(lending_club_df)
    plot_loan_purpose_performance(purpose_df)

    anomalies = detect_anomalies(forecast_results["monthly_kpis"])
    plot_anomaly_detection(forecast_results["monthly_kpis"], anomalies)

    recommendations = generate_recommendations(purpose_df, model_results["metrics"], forecast_results)
    print_recommendations(recommendations)

    html = build_html_report(model_results["metrics"], purpose_df, forecast_results, recommendations)
    save_report(html)

    print(" Pipeline Complete!")
    print(" Report: reports/output/business_performance_report.html")
    print(" Dashboard: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
