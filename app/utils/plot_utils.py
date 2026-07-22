"""
Plot Utilities
==============

Small collection of pure plotting functions shared by agents that need to
generate figures (EDA, Evaluation, Explainability). Kept intentionally
minimal — this is NOT a "tools" abstraction layer, just deduplicated code
for the one thing that genuinely repeats across agents: saving matplotlib
figures to disk and returning their path.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

OUTPUT_DIR = os.path.join("data", "outputs", "plots")


def ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def save_current_figure(filename: str) -> str:
    """Saves the current matplotlib figure and returns its path."""
    output_dir = ensure_output_dir()
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_correlation_heatmap(df: pd.DataFrame, numerical_columns: List[str]) -> str | None:
    numeric_df = df[numerical_columns]
    if numeric_df.shape[1] < 2:
        return None
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    return save_current_figure("correlation_matrix.png")

def plot_confusion_matrix(cm, class_labels=None) -> str:
    import numpy as np
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=class_labels)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    return save_current_figure("confusion_matrix.png")


def plot_roc_curve(y_true, y_proba) -> str:
    plt.figure(figsize=(6, 5))
    try:
        RocCurveDisplay.from_predictions(y_true, y_proba)
    except Exception:
        from sklearn.preprocessing import LabelEncoder
        y_encoded = LabelEncoder().fit_transform(y_true)
        RocCurveDisplay.from_predictions(y_encoded, y_proba)
    plt.title("ROC Curve")
    return save_current_figure("roc_curve.png")


def plot_missing_values(df: pd.DataFrame) -> str | None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return None
    plt.figure(figsize=(8, max(4, len(missing) * 0.4)))
    sns.barplot(x=missing.values, y=missing.index, orient="h")
    plt.title("Missing Values per Column")
    plt.xlabel("Missing count")
    return save_current_figure("missing_values.png")


def plot_distribution(df: pd.DataFrame, column: str) -> str:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[column].dropna(), kde=True)
    plt.title(f"Distribution: {column}")
    return save_current_figure(f"distribution_{column}.png")


def plot_target_balance(df: pd.DataFrame, target_column: str, problem_type: str) -> str:
    plt.figure(figsize=(6, 4))
    if problem_type == "classification":
        sns.countplot(x=target_column, data=df)
    else:
        sns.histplot(df[target_column].dropna(), kde=True)
    plt.title(f"Target Distribution: {target_column}")
    return save_current_figure("target_distribution.png")


def detect_outliers_iqr(df: pd.DataFrame, numerical_columns: List[str]) -> Dict[str, int]:
    """Returns {column: outlier_count} using the classic 1.5*IQR rule."""
    outlier_counts: Dict[str, int] = {}
    for col in numerical_columns:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        if count > 0:
            outlier_counts[col] = count
    return outlier_counts