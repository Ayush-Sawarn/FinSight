"""
dashboard.py
------------
Interactive Streamlit dashboard for FinSight.
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys
import os
import warnings

warnings.filterwarnings("ignore")
sys.path.append(".")

# ── Page Config ──
st.set_page_config(
    page_title="FinSight — Business Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        border-radius: 12px; padding: 18px; text-align: center;
    }
    .stMetric { background: #F0F4FF; border-radius: 10px; padding: 12px; }
    h1 { color: #1565C0 !important; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]


# ── Data Loading ──
@st.cache_data
def load_data():
    try:
        from data.data_loader import load_raw_data, clean_data
        df = load_raw_data()
        df_clean = clean_data(df)
        return df, df_clean
    except FileNotFoundError:
        st.warning("Dataset not found. Using synthetic demo data.")
        return _generate_demo_data()


def _generate_demo_data():
    """Generate synthetic data for demo purposes."""
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        "Age": np.random.randint(21, 70, n),
        "Income": np.random.randint(200000, 2000000, n),
        "LoanAmount": np.random.randint(100000, 1500000, n),
        "CreditScore": np.random.randint(300, 850, n),
        "LoanTenure": np.random.choice([12, 24, 36, 48, 60], n),
        "Default": np.random.choice([0, 1], n, p=[0.80, 0.20]),
        "Employment": np.random.choice(["Salaried", "Self-Employed", "Business"], n),
        "Purpose": np.random.choice(["Home", "Education", "Medical", "Business", "Personal"], n),
    })
    return df, df


@st.cache_data
def get_monthly_kpis():
    from models.forecasting import build_monthly_kpis, load_lending_club_data
    return build_monthly_kpis(load_lending_club_data())


@st.cache_data
def get_purpose_data():
    from models.forecasting import load_lending_club_data
    from utils.insights import analyze_loan_purpose
    return analyze_loan_purpose(load_lending_club_data())


# ────────────────────────────
# SIDEBAR
# ────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1565C0/FFFFFF?text=FinSight", width=200)
    st.markdown("### Filters")
    age_range = st.slider("Age Range", 18, 80, (21, 65))
    default_filter = st.selectbox("Loan Status", ["All", "Non-Default (0)", "Default (1)"])
    st.markdown("---")
    st.markdown("**FinSight v1.0**")
    st.markdown("Consumer Finance Analytics")


# ────────────────────────────
# MAIN CONTENT
# ────────────────────────────
st.title("FinSight — Business Performance Dashboard")
st.markdown("*Loan portfolio analytics, forecasting, and decision support*")
st.markdown("---")

# Load data
df_raw, df_clean = load_data()

# Apply filters
df_filtered = df_raw.copy()
if "Age" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["Age"] >= age_range[0]) &
        (df_filtered["Age"] <= age_range[1])
    ]
if default_filter == "Non-Default (0)":
    df_filtered = df_filtered[df_filtered["Default"] == 0]
elif default_filter == "Default (1)":
    df_filtered = df_filtered[df_filtered["Default"] == 1]

# ── KPI Metrics Row ──
col1, col2, col3, col4, col5 = st.columns(5)
total = len(df_filtered)
defaults = df_filtered["Default"].sum() if "Default" in df_filtered.columns else 0
default_rate = defaults / total * 100 if total > 0 else 0
avg_loan = df_filtered["LoanAmount"].mean() if "LoanAmount" in df_filtered.columns else 0
avg_income = df_filtered["Income"].mean() if "Income" in df_filtered.columns else 0

col1.metric("Total Records", f"{total:,}", delta=None)
col2.metric("Total Defaults", f"{defaults:,}")
col3.metric("Default Rate", f"{default_rate:.2f}%", delta=f"{default_rate - 20:.1f}% vs 20% target")
col4.metric("Avg Loan Amount", f"{avg_loan:,.0f}")
col5.metric("Avg Income", f"{avg_income:,.0f}")

st.markdown("---")

# ── Tabs ──
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Forecasting", "Loan Purpose",
    "Model", "Insights"
])


# ── TAB 1: Overview ──
with tab1:
    st.subheader("Loan Portfolio Overview")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Default Distribution**")
        if "Default" in df_filtered.columns:
            counts = df_filtered["Default"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(["Non-Default", "Default"], counts.reindex([0, 1]).fillna(0).values,
                   color=PALETTE[:2], edgecolor="white", width=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close()

    with col_b:
        st.markdown("**Default Rate by Age Group**")
        if "Age" in df_filtered.columns and "Default" in df_filtered.columns:
            df_filtered["age_group"] = pd.cut(
                df_filtered["Age"], bins=[18, 25, 35, 45, 55, 100],
                labels=["18-25", "26-35", "36-45", "46-55", "55+"]
            )
            rate_by_age = df_filtered.groupby("age_group", observed=True)["Default"].mean() * 100
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(rate_by_age.index.astype(str), rate_by_age.values,
                    marker="o", color=PALETTE[0], linewidth=2.5)
            ax.fill_between(range(len(rate_by_age)), rate_by_age.values,
                            alpha=0.12, color=PALETTE[0])
            ax.set_ylabel("Default Rate (%)")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)
            plt.close()

    st.markdown("**Loan Amount Distribution by Default Status**")
    if "LoanAmount" in df_filtered.columns and "Default" in df_filtered.columns:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for val, label, color in zip([0, 1], ["Non-Default", "Default"], PALETTE[:2]):
            subset = df_filtered[df_filtered["Default"] == val]["LoanAmount"].dropna()
            ax.hist(subset, bins=50, alpha=0.65, label=label, color=color, edgecolor="white")
        ax.set_xlabel("Loan Amount")
        ax.set_ylabel("Count")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
        st.pyplot(fig)
        plt.close()

    st.markdown("**Raw Data Sample**")
    st.dataframe(df_filtered.head(100), use_container_width=True)


# ── TAB 2: Forecasting ──
with tab2:
    st.subheader("6-Month Business Forecasting")

    try:
        monthly = get_monthly_kpis()
        from models.forecasting import arima_forecast

        kpi_choice = st.selectbox(
            "Select KPI to Forecast",
            ["total_loans_disbursed", "default_rate_%", "loan_count"]
        )

        series = monthly[kpi_choice]
        forecast = arima_forecast(series, steps=6)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(series.index, series.values, color=PALETTE[0],
                linewidth=2, label="Historical", marker="o", markersize=3)
        ax.plot(forecast.index, forecast.values, color=PALETTE[1],
                linewidth=2.5, linestyle="--", label="Forecast", marker="s", markersize=5)
        ax.fill_between(forecast.index,
                        forecast.values * 0.95, forecast.values * 1.05,
                        alpha=0.15, color=PALETTE[1], label="±5% Band")
        ax.axvline(x=series.index[-1], color="gray", linestyle=":", linewidth=1.2)
        ax.set_title(f"{kpi_choice.replace('_', ' ').title()} — 6-Month Forecast",
                     fontsize=13, fontweight="bold")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

        st.markdown("**Forecast Values**")
        fc_df = forecast.reset_index()
        fc_df.columns = ["Month", "Forecasted Value"]
        fc_df["Month"] = fc_df["Month"].dt.strftime("%B %Y")
        fc_df["Forecasted Value"] = fc_df["Forecasted Value"].round(4)
        st.dataframe(fc_df, use_container_width=True)

    except Exception as e:
        st.error(f"Forecasting error: {e}")


# ── TAB 3: Loan Purpose ──
with tab3:
    st.subheader("Loan Purpose Performance (Lending Club)")

    try:
        purpose_df = get_purpose_data()

        col_x, col_y = st.columns(2)

        with col_x:
            st.markdown("**Loan Count by Purpose**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.barh(purpose_df["Purpose"], purpose_df["Total_Loans"],
                    color=PALETTE, edgecolor="white", alpha=0.85)
            ax.invert_yaxis()
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)
            plt.close()

        with col_y:
            st.markdown("**Resolved Default Rate by Purpose**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(purpose_df["Purpose"], purpose_df["Default_Rate_%"] * 100,
                   color=PALETTE[1], edgecolor="white", alpha=0.85)
            ax.set_ylabel("Default Rate (%)")
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)
            plt.close()

        st.markdown("**Purpose Summary**")
        display_df = purpose_df.copy()
        display_df["Default_Rate_%"] = display_df["Default_Rate_%"].map(lambda value: f"{value:.1%}" if pd.notna(value) else "N/A")
        st.dataframe(display_df, use_container_width=True)

    except Exception as e:
        st.error(f"Loan purpose analysis error: {e}")


# ── TAB 4: Model ──
with tab4:
    st.subheader("Default Prediction Model")
    st.info("Train the model first by running: `python models/default_predictor.py`")

    model_path = "models/saved/random_forest.pkl"
    if os.path.exists(model_path):
        import joblib
        model = joblib.load(model_path)
        st.success("✅ Random Forest model loaded")

        st.markdown("**Predict Default Risk for a New Applicant**")
        c1, c2, c3 = st.columns(3)
        age_in = c1.number_input("Age", 18, 80, 35)
        income_in = c2.number_input("Annual Income", 100000, 5000000, 600000, step=50000)
        loan_in = c3.number_input("Loan Amount", 50000, 3000000, 500000, step=50000)

        if st.button("🔍 Predict Default Risk"):
            lti = loan_in / (income_in + 1)
            input_df = pd.DataFrame([[age_in, income_in, loan_in, lti]],
                                    columns=["Age", "Income", "LoanAmount", "loan_to_income_ratio"])
            try:
                proba = model.predict_proba(input_df)[0][1]
                risk_label = "🔴 HIGH RISK" if proba > 0.5 else "🟢 LOW RISK"
                st.metric("Default Probability", f"{proba*100:.1f}%")
                st.markdown(f"**Risk Level: {risk_label}**")
            except Exception as e:
                st.error(f"Prediction error: {e}\nRetrain model with matching feature set.")
    else:
        st.warning("Model not trained yet. Run: `python models/default_predictor.py`")


# ── TAB 5: Insights ──
with tab5:
    st.subheader("Business Insights & Recommendations")

    try:
        purpose_df = get_purpose_data()
        from utils.insights import generate_recommendations
        recs = generate_recommendations(purpose_df, None, {})

        priority_colors = {"HIGH": "#F44336", "MEDIUM": "#FF9800", "INFO": "#2196F3"}

        for rec in recs:
            color = priority_colors.get(rec["priority"], "#9E9E9E")
            st.markdown(f"""
            <div style="border-left:5px solid {color}; background:#FAFAFA;
                        padding:14px 18px; border-radius:8px; margin-bottom:14px;">
                <span style="background:{color};color:white;padding:3px 10px;
                             border-radius:10px;font-size:0.75em;font-weight:700;">
                    {rec['priority']}
                </span>
                <strong style="margin-left:8px;">{rec['area']}</strong>
                <p style="margin-top:8px;color:#555;font-size:0.9em;">
                    <em>Insight:</em> {rec['insight']}</p>
                <p style="color:#555;font-size:0.9em;">
                    <em>Action:</em> {rec['action']}</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Insights error: {e}")
