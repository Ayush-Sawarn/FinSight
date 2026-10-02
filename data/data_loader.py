# Dataset: https://www.kaggle.com/datasets/nikhil1e9/loan-default


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


def load_raw_data(filepath: str = "data/loan_default.csv") -> pd.DataFrame:
    """Load raw CSV into a DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}.\n"
            "Download from: https://www.kaggle.com/datasets/nikhil1e9/loan-default\n"
            "and place loan_default.csv inside the data/ folder."
        )
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess raw loan data.
    Steps:
      1. Drop duplicates
      2. Handle missing values
      3. Encode categorical columns
      4. Feature engineering
    """
    df = df.copy()

    # 1. Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Dropped {before - len(df)} duplicate rows")

    # 2. Handle missing values
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    print(f"Imputed missing values — numeric: median, categorical: mode")

    # 3. Encode categorical columns
    le = LabelEncoder()
    for col in cat_cols:
        if col != "Default":  # keep target readable
            df[col] = le.fit_transform(df[col].astype(str))

    # 4. Feature engineering
    if "Income" in df.columns and "LoanAmount" in df.columns:
        df["loan_to_income_ratio"] = df["LoanAmount"] / (df["Income"] + 1)

    if "Age" in df.columns:
        df["age_group"] = pd.cut(
            df["Age"],
            bins=[0, 25, 35, 50, 100],
            labels=["<25", "25-35", "35-50", "50+"],
        )
        df["age_group"] = LabelEncoder().fit_transform(df["age_group"].astype(str))

    print(f"Feature engineering complete — shape: {df.shape}")
    return df


def get_processed_data(filepath: str = "data/loan_default.csv"):
    """Full pipeline: load → clean → return train/test split."""
    from sklearn.model_selection import train_test_split

    df = load_raw_data(filepath)
    df = clean_data(df)

    target_col = "Default"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_processed_data()
    print(X_train.head())
