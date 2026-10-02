# FinSight: Business Performance Analytics & Forecasting Platform

FinSight is a portfolio project demonstrating consumer finance analytics,
credit-risk modeling, KPI forecasting, performance reporting, and decision
support. The existing loan default CSV is used for default classification;
Lending Club accepted-loan records provide the dated history for real monthly
portfolio forecasts and loan-purpose analysis.

## Dataset

**Loan Default Prediction Dataset** (Kaggle)
- Link: https://www.kaggle.com/datasets/nikhil1e9/loan-default
- 255,347 rows | 18 columns
- Target: `Default` (0 = paid, 1 = defaulted)

**Lending Club Accepted Loans** (Kaggle)
- Download the accepted-loan file from https://www.kaggle.com/datasets/wordsforthewise/lending-club
- Place `accepted_2007_to_2018Q4.csv.gz` in `data/` (leave it gzip-compressed).
- The forecast and loan-purpose analysis use `issue_d`, `loan_status`, `funded_amnt` or `loan_amnt`, and `purpose`.
- `loan_default.csv` remains the input to the classification model; the two datasets are not merged.

### Setup
1. Download `loan_default.csv` and place it in `data/`.
2. Download the Lending Club accepted-loan CSV and place it at `data/accepted_2007_to_2018Q4.csv.gz`.

## Project Structure

```
FinSight/
├── data/
│   └── data_loader.py       # Data loading, cleaning, feature engineering
├── models/
│   ├── forecasting.py       # ARIMA time-series forecasting
│   └── default_predictor.py # Random Forest + XGBoost classification
├── utils/
│   ├── eda.py               # Exploratory data analysis & charts
│   └── insights.py          # Channel analysis + recommendation engine
├── reports/
│   ├── generate_report.py   # HTML business performance report
│   └── figures/             # Auto-generated charts
├── dashboard.py             # Streamlit interactive dashboard
├── main.py                  # Master pipeline runner
└── requirements.txt
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (EDA + Forecasting + Model + Report)
python main.py

# Launch interactive dashboard
streamlit run dashboard.py
```

## What It Does

| Module | Description |
|---|---|
| `data_loader.py` | Loads, cleans, engineers features from loan dataset |
| `eda.py` | Default distribution, age analysis, correlation heatmap |
| `forecasting.py` | Aggregates Lending Club loans by issue month and ARIMA-forecasts loan volume, loan count, and resolved vintage default rate |
| `default_predictor.py` | RF + XGBoost model with ROC-AUC, F1, feature importance |
| `insights.py` | Loan-purpose summaries, monthly anomaly detection, and rule-based recommendations |
| `generate_report.py` | Auto-generates HTML business performance report |
| `dashboard.py` | Streamlit app with loan analysis, forecasts, purpose summaries, and risk prediction |

## Data Scope

The 255K-row loan default CSV supports exploratory analysis and classification,
but has no dates. Lending Club supplies issue dates and outcomes for monthly
loan-count, funded-volume, and resolved-default-rate forecasts. Default rates
are calculated only from loans labeled Fully Paid, Charged Off, or Default;
ongoing and late loans are excluded. This is a vintage default rate by issue
month, not a calendar-month count of defaults. Lending Club has no reliable
distribution-channel field, so this version analyzes loan purpose instead.

## Project Highlights

- Built a loan-default analysis pipeline with data cleaning, exploratory analysis,
  model training, evaluation, and reporting.
- Added six-month ARIMA forecasts based on Lending Club's actual monthly loan
  originations and resolved outcomes.
- Created a Streamlit dashboard and HTML report to present model metrics, charts,
  anomalies, and rule-based recommendations.
