"""
default_predictor.py
---------------------
Trains and evaluates a loan default prediction model.
Models: Random Forest + XGBoost with cross-validation.
Outputs: model metrics, feature importance chart, ROC curve.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import joblib
import warnings

warnings.filterwarnings("ignore")
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models/saved", exist_ok=True)

PALETTE = ["#246b68", "#b85c50", "#577590", "#c28e3e"]


# ─────────────────────────────────────────────
# 1. Train Models
# ─────────────────────────────────────────────
def train_models(X_train, y_train) -> dict:
    """Train Random Forest and attempt XGBoost."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    models = {}

    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    rf_cv = cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"Random Forest — CV ROC-AUC: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
    models["random_forest"] = {"model": rf, "cv_auc": rf_cv.mean()}
    joblib.dump(rf, "models/saved/random_forest.pkl")

    # XGBoost (optional)
    try:
        from xgboost import XGBClassifier
        print("Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        xgb.fit(X_train, y_train)
        xgb_cv = cross_val_score(xgb, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
        print(f"XGBoost — CV ROC-AUC: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")
        models["xgboost"] = {"model": xgb, "cv_auc": xgb_cv.mean()}
        joblib.dump(xgb, "models/saved/xgboost.pkl")
    except ImportError:
        print("[!] XGBoost not installed — skipping (pip install xgboost)")

    return models


# ─────────────────────────────────────────────
# 2. Evaluate Models
# ─────────────────────────────────────────────
def evaluate_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate all trained models and return metrics DataFrame."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )

    rows = []
    for name, obj in models.items():
        model = obj["model"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        rows.append({
            "Model": name.replace("_", " ").title(),
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        })

    metrics_df = pd.DataFrame(rows).set_index("Model")
    print("\n── Model Evaluation Metrics ──")
    print(metrics_df.to_string())
    return metrics_df


# ─────────────────────────────────────────────
# 3. Feature Importance
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, model_name: str = "Random Forest"):
    """Bar chart of top-15 most important features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(range(len(indices)),
                   importances[indices],
                   color=PALETTE[0], alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_title(f"Top 15 Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.spines[["top", "right"]].set_visible(False)

    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}", va="center", fontsize=8)

    plt.tight_layout()
    fname = f"reports/figures/feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────
# 4. ROC Curves
# ─────────────────────────────────────────────
def plot_roc_curves(models: dict, X_test, y_test):
    """Overlay ROC curves for all models."""
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (name, obj) in enumerate(models.items()):
        model = obj["model"]
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[i],
                linewidth=2.5, label=f"{name.replace('_', ' ').title()} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Default Prediction Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("reports/figures/roc_curves.png")
    plt.close()
    print("Saved: reports/figures/roc_curves.png")


# ─────────────────────────────────────────────
# 5. Confusion Matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(model, X_test, y_test, model_name: str = "Random Forest"):
    """Heatmap confusion matrix."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Non-Default", "Default"],
                yticklabels=["Non-Default", "Default"],
                linewidths=0.5)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    fname = f"reports/figures/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────
# 6. Run Full Pipeline
# ─────────────────────────────────────────────
def run_default_prediction_pipeline(X_train, X_test, y_train, y_test) -> dict:
    print("\n═══ Running Default Prediction Pipeline ═══")
    models = train_models(X_train, y_train)
    metrics = evaluate_models(models, X_test, y_test)

    best_model_name = metrics["ROC-AUC"].idxmax().lower().replace(" ", "_")
    best_model = models[best_model_name]["model"]

    plot_feature_importance(best_model, list(X_train.columns), best_model_name.replace("_", " ").title())
    plot_roc_curves(models, X_test, y_test)
    plot_confusion_matrix(best_model, X_test, y_test, best_model_name.replace("_", " ").title())

    print(f"\nBest model: {best_model_name} — ROC-AUC: {metrics.loc[best_model_name.replace('_',' ').title(), 'ROC-AUC']}")
    return {"models": models, "metrics": metrics, "best_model": best_model}


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from data.data_loader import get_processed_data
    X_train, X_test, y_train, y_test = get_processed_data()
    results = run_default_prediction_pipeline(X_train, X_test, y_train, y_test)
