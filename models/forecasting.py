"""Monthly loan portfolio forecasting from Lending Club loan records."""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.makedirs("reports/figures", exist_ok=True)

PALETTE = ["#246b68", "#b85c50", "#577590"]
DEFAULT_LENDING_CLUB_PATH = "data/accepted_2007_to_2018Q4.csv.gz"


def load_lending_club_data(filepath: str = DEFAULT_LENDING_CLUB_PATH) -> pd.DataFrame:
    """Load Lending Club's accepted-loan CSV (plain CSV or gzip-compressed)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Lending Club data not found at {filepath}. Download the accepted-loan "
            "CSV from https://www.kaggle.com/datasets/wordsforthewise/lending-club "
            "and place it at this path."
        )
    wanted = {"issue_d", "loan_status", "funded_amnt", "loan_amnt", "purpose"}
    return pd.read_csv(
        filepath,
        usecols=lambda column: column in wanted,
        low_memory=False,
    )


def build_monthly_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate actual Lending Club loans into monthly origination metrics.

    Default rate is calculated by origination month, using only loans with a
    resolved outcome (Fully Paid, Charged Off, or Default). Current and late
    loans are excluded from that rate because their final outcome is unknown.
    """
    required = {"issue_d", "loan_status"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Lending Club data is missing required columns: {', '.join(sorted(missing))}")

    amount_col = "funded_amnt" if "funded_amnt" in df.columns else "loan_amnt"
    if amount_col not in df.columns:
        raise ValueError("Lending Club data must include funded_amnt or loan_amnt.")

    work = df[["issue_d", "loan_status", amount_col]].copy()
    issue_dates = pd.to_datetime(work["issue_d"].astype(str).str.strip(), format="%b-%Y", errors="coerce")
    fallback_dates = pd.to_datetime(work["issue_d"], errors="coerce")
    work["issue_date"] = issue_dates.fillna(fallback_dates)
    work["loan_amount"] = pd.to_numeric(work[amount_col], errors="coerce")
    status = work["loan_status"].astype(str).str.strip().str.lower()

    paid = status.str.contains("fully paid", regex=False)
    defaulted = status.str.contains("charged off", regex=False) | status.eq("default")
    work["resolved_default"] = np.select([defaulted, paid], [1.0, 0.0], default=np.nan)
    work = work.dropna(subset=["issue_date", "loan_amount"])
    if work.empty:
        raise ValueError("No rows had a usable issue_d date and loan amount.")

    work["issue_month"] = work["issue_date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = work.groupby("issue_month").agg(
        loan_count=("loan_status", "size"),
        total_loans_disbursed=("loan_amount", "sum"),
        default_rate=("resolved_default", "mean"),
        total_defaults=("resolved_default", "sum"),
        resolved_loans=("resolved_default", "count"),
    ).rename(columns={"default_rate": "default_rate_%"})

    all_months = pd.date_range(monthly.index.min(), monthly.index.max(), freq="ME")
    monthly = monthly.reindex(all_months)
    monthly.index.name = "date"
    monthly["loan_count"] = monthly["loan_count"].fillna(0)
    monthly["total_loans_disbursed"] = monthly["total_loans_disbursed"].fillna(0)
    monthly["total_defaults"] = monthly["total_defaults"].fillna(0)
    monthly["resolved_loans"] = monthly["resolved_loans"].fillna(0)

    print(f"Built monthly portfolio history from {len(work):,} usable loan records ({len(monthly)} months).")
    print("Monthly default rates include resolved loans only; unresolved loans are excluded.")
    return monthly


def arima_forecast(series: pd.Series, steps: int = 6) -> pd.Series:
    """Fit ARIMA(2,1,2), falling back to a linear trend if statsmodels is absent."""
    series = series.dropna().astype(float)
    if len(series) < 3:
        raise ValueError(f"At least 3 observed values are needed to forecast {series.name!r}.")
    try:
        from statsmodels.tsa.arima.model import ARIMA

        result = ARIMA(series, order=(2, 1, 2)).fit()
        forecast = result.forecast(steps=steps)
    except ImportError:
        x = np.arange(len(series))
        coefficient = np.polyfit(x, series.values, 1)
        forecast = pd.Series(
            np.polyval(coefficient, np.arange(len(series), len(series) + steps)),
            index=pd.date_range(series.index[-1] + pd.offsets.MonthEnd(1), periods=steps, freq="ME"),
            name=series.name,
        )
    forecast.index = pd.date_range(series.index[-1] + pd.offsets.MonthEnd(1), periods=steps, freq="ME")
    forecast.name = series.name
    return forecast


def backtest_arima(series: pd.Series, horizon: int = 6) -> dict:
    """Evaluate ARIMA on the final horizon, which is held out from fitting.

    Forecasts are recursive multi-step predictions made from the earlier
    observations only. MAPE ignores zero actuals, where percentage error is
    undefined. Returns NaN metrics if there is not enough history.
    """
    observed = series.dropna().astype(float)
    minimum_train_size = 24
    result = {
        "MAE": np.nan,
        "RMSE": np.nan,
        "MAPE (%)": np.nan,
        "Backtest Months": 0,
    }
    if horizon < 1 or len(observed) < minimum_train_size + horizon:
        return result

    train = observed.iloc[:-horizon]
    actual = observed.iloc[-horizon:]
    prediction = arima_forecast(train, steps=horizon).to_numpy(dtype=float)

    # Apply the same valid-range constraints used for the production forecast.
    if "rate" in str(series.name).lower():
        prediction = np.clip(prediction, 0.0, 1.0)
    else:
        prediction = np.clip(prediction, 0.0, None)

    actual_values = actual.to_numpy(dtype=float)
    metric_scale = 100.0 if "rate" in str(series.name).lower() else 1.0
    errors = (prediction - actual_values) * metric_scale
    nonzero = actual_values != 0
    result["MAE"] = float(np.mean(np.abs(errors)))
    result["RMSE"] = float(np.sqrt(np.mean(np.square(errors))))
    # MAPE is unstable for default-rate series when actuals are close to zero.
    # Keep absolute errors in percentage points for that KPI and omit MAPE.
    is_rate = "rate" in str(series.name).lower()
    result["MAPE (%)"] = (
        float(np.mean(np.abs(errors[nonzero] / (actual_values[nonzero] * metric_scale))) * 100)
        if nonzero.any() and not is_rate else np.nan
    )
    result["Backtest Months"] = int(len(actual_values))
    return result


def plot_forecast(series: pd.Series, forecast: pd.Series, title: str,
                  ylabel: str, filename: str, percentage: bool = False):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(series.index, series.values, color=PALETTE[0], linewidth=2,
            label="Historical", marker="o", markersize=3)
    ax.plot(forecast.index, forecast.values, color=PALETTE[1], linewidth=2.5,
            linestyle="--", label="Forecast", marker="s", markersize=5)
    ax.axvline(x=series.index[-1], color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
    if percentage:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    else:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"reports/figures/{filename}")
    plt.close()


def run_forecasting(df: pd.DataFrame) -> dict:
    """Forecast monthly originated loan count, funded amount, and vintage default rate."""
    monthly = build_monthly_kpis(df)
    results = {"monthly_kpis": monthly}
    forecast_specs = [
        ("total_loans_disbursed", "loans_disbursed_forecast", "Loan Volume", "Loan Amount", "forecast_loans_disbursed.png", False),
        ("loan_count", "loan_count_forecast", "Loan Originations", "Number of Loans", "forecast_new_customers.png", False),
        ("default_rate_%", "default_rate_forecast", "Resolved Default Rate by Origination Month", "Default Rate", "forecast_default_rate.png", True),
    ]

    for column, result_key, title, ylabel, filename, percentage in forecast_specs:
        history = monthly[column].dropna()
        backtest = backtest_arima(history, horizon=6)
        backtest["KPI"] = title
        backtest["MAE Unit"] = "percentage points" if percentage else ylabel
        results.setdefault("forecast_metrics", []).append(backtest)
        forecast = arima_forecast(history, steps=6)
        if percentage:
            forecast = forecast.clip(0, 1)
        else:
            forecast = forecast.clip(lower=0)
        plot_forecast(history, forecast, f"{title} — 6-Month Forecast", ylabel, filename, percentage)
        results[result_key] = forecast

    results["forecast_metrics"] = pd.DataFrame(results["forecast_metrics"]).set_index("KPI")
    print("\nARIMA six-month holdout backtest (last six observed months):")
    print(results["forecast_metrics"].round(3).to_string())

    return results


if __name__ == "__main__":
    lending_club = load_lending_club_data()
    forecast_results = run_forecasting(lending_club)
    print(forecast_results["loans_disbursed_forecast"].round(0))
