"""
eda.py
------
Exploratory Data Analysis utilities.
Generates charts and summary statistics used in the business performance report.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

os.makedirs("reports/figures", exist_ok=True)

PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 120})


# ─────────────────────────────────────────────
# 1. Dataset Summary
# ─────────────────────────────────────────────
def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with per-column dtype, missing %, and unique counts."""
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "missing_%": (df.isnull().sum() / len(df) * 100).round(2),
        "unique_values": df.nunique(),
        "sample_value": df.iloc[0],
    })
    print("\n── Dataset Summary ──")
    print(summary.to_string())
    return summary


# ─────────────────────────────────────────────
# 2. Target Distribution
# ─────────────────────────────────────────────
def plot_target_distribution(df: pd.DataFrame, target_col: str = "Default"):
    """Bar chart of default vs non-default."""
    counts = df[target_col].value_counts()
    labels = ["Non-Default", "Default"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=PALETTE[:2], edgecolor="white", width=0.5)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=10)

    ax.set_title("Loan Default Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("reports/figures/target_distribution.png")
    plt.close()
    print("[✓] Saved: reports/figures/target_distribution.png")


# ─────────────────────────────────────────────
# 3. Numeric Feature Distributions
# ─────────────────────────────────────────────
def plot_numeric_distributions(df: pd.DataFrame, cols: list = None):
    """Histograms for numeric columns."""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()[:8]

    n = len(cols)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=40, color=PALETTE[i % len(PALETTE)],
                     edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=10, fontweight="bold")
        axes[i].spines[["top", "right"]].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("reports/figures/numeric_distributions.png", bbox_inches="tight")
    plt.close()
    print("[✓] Saved: reports/figures/numeric_distributions.png")


# ─────────────────────────────────────────────
# 4. Default Rate by Age Group
# ─────────────────────────────────────────────
def plot_default_rate_by_age(df: pd.DataFrame):
    """Line chart: default rate across age buckets."""
    if "Age" not in df.columns or "Default" not in df.columns:
        print("[!] Age or Default column not found — skipping age plot")
        return

    df = df.copy()
    df["age_bin"] = pd.cut(df["Age"], bins=[18, 25, 35, 45, 55, 100],
                           labels=["18-25", "26-35", "36-45", "46-55", "55+"])
    rate = df.groupby("age_bin", observed=True)["Default"].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rate.index.astype(str), rate.values, marker="o",
            color=PALETTE[0], linewidth=2.5, markersize=8)
    ax.fill_between(range(len(rate)), rate.values, alpha=0.1, color=PALETTE[0])

    for i, (x, y) in enumerate(zip(range(len(rate)), rate.values)):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)

    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels(rate.index.astype(str))
    ax.set_title("Default Rate by Age Group", fontsize=14, fontweight="bold")
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("Age Group")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("reports/figures/default_rate_by_age.png")
    plt.close()
    print("[✓] Saved: reports/figures/default_rate_by_age.png")


# ─────────────────────────────────────────────
# 5. Correlation Heatmap
# ─────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    """Heatmap of feature correlations."""
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/figures/correlation_heatmap.png")
    plt.close()
    print("[✓] Saved: reports/figures/correlation_heatmap.png")


# ─────────────────────────────────────────────
# 6. Loan Amount Distribution by Default
# ─────────────────────────────────────────────
def plot_loan_amount_by_default(df: pd.DataFrame):
    """Box plot comparing loan amounts for defaulters vs non-defaulters."""
    loan_col = next((c for c in df.columns if "loan" in c.lower() and "amount" in c.lower()), None)
    if loan_col is None:
        print("[!] Loan amount column not found — skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    groups = [df[df["Default"] == 0][loan_col].dropna(),
              df[df["Default"] == 1][loan_col].dropna()]
    bp = ax.boxplot(groups, patch_artist=True, notch=True,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE[:2]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticklabels(["Non-Default", "Default"])
    ax.set_title("Loan Amount Distribution by Default Status", fontsize=13, fontweight="bold")
    ax.set_ylabel("Loan Amount")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("reports/figures/loan_amount_by_default.png")
    plt.close()
    print("[✓] Saved: reports/figures/loan_amount_by_default.png")


# ─────────────────────────────────────────────
# Run all EDA
# ─────────────────────────────────────────────
def run_full_eda(df: pd.DataFrame):
    print("\n═══ Running Full EDA ═══")
    dataset_summary(df)
    plot_target_distribution(df)
    plot_numeric_distributions(df)
    plot_default_rate_by_age(df)
    plot_correlation_heatmap(df)
    plot_loan_amount_by_default(df)
    print("\n[✓] EDA complete — all figures saved to reports/figures/")


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from data.data_loader import load_raw_data
    df = load_raw_data()
    run_full_eda(df)
