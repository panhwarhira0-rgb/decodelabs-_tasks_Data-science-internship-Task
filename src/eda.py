"""
eda.py
------
All Exploratory Data Analysis functions.
Every plot is saved to disk — never just displayed in a notebook.
This lets the pipeline run headlessly on a server.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from src.config import PLOTS_DIR, FIGURE_DPI, PALETTE, NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL


def _save(fig: plt.Figure, filename: str) -> None:
    """Save a matplotlib figure to the plots directory."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot saved]  {filename}")


def initial_inspection(df: pd.DataFrame) -> dict:
    """
    Print and return a structured summary of the dataset.
    This is the very first thing you do — understand WHAT you have.
    """
    print("\n" + "="*60)
    print("  INITIAL DATASET INSPECTION")
    print("="*60)
    print(f"\nShape          : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn Names   :\n{list(df.columns)}")
    print(f"\nData Types     :\n{df.dtypes}")
    print(f"\nDuplicate Rows : {df.duplicated().sum()}")
    print(f"\nFirst 5 Rows   :\n{df.head()}")
    print(f"\nStatistical Summary (Numerical):\n{df.describe().round(2)}")

    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "missing_total": int(df.isnull().sum().sum()),
    }


def plot_missing_values(df: pd.DataFrame) -> None:
    """
    Visualize missing values as a percentage bar chart.
    WHY: You need to SEE the missing pattern, not just read numbers.
    """
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if missing_pct.empty:
        print("  No missing values found — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(missing_pct.index, missing_pct.values, color=sns.color_palette(PALETTE, len(missing_pct)))
    ax.axhline(y=5,  color="green",  linestyle="--", linewidth=1.2, label="5% threshold (drop rows)")
    ax.axhline(y=20, color="orange", linestyle="--", linewidth=1.2, label="20% threshold (KNN)")
    ax.set_title("Missing Value Percentage per Column", fontsize=14, fontweight="bold")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Missing %")
    ax.legend()
    for bar, val in zip(bars, missing_pct.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    _save(fig, "01_missing_values.png")


def plot_distributions(df: pd.DataFrame) -> None:
    """
    Histograms with KDE for all numerical columns.
    WHY: Tells us about shape (normal vs skewed) → informs imputation strategy.
    """
    cols = [c for c in NUMERICAL_COLS if c in df.columns]
    n = len(cols)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color=sns.color_palette(PALETTE)[i % 6])
        axes[i].set_title(f"Distribution: {col}", fontsize=11)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        skew_val = df[col].skew()
        axes[i].text(0.97, 0.95, f"Skew: {skew_val:.2f}", transform=axes[i].transAxes,
                     ha="right", va="top", fontsize=8, color="darkred")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Numerical Feature Distributions", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(fig, "02_distributions.png")


def plot_boxplots(df: pd.DataFrame) -> None:
    """
    Boxplots to visualize outlier presence BEFORE treatment.
    WHY: The whiskers of a boxplot show the IQR fence — outliers appear as dots beyond them.
    """
    cols = [c for c in NUMERICAL_COLS if c in df.columns]
    fig, axes = plt.subplots(2, (len(cols) + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.boxplot(y=df[col].dropna(), ax=axes[i], color=sns.color_palette(PALETTE)[i % 6])
        axes[i].set_title(f"Boxplot: {col}", fontsize=11)
        axes[i].set_ylabel(col)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Boxplots — Outlier Visualization (Before Treatment)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03_boxplots_before.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Correlation heatmap for numerical features.
    WHY: Reveals multicollinearity — two features carrying the same info hurt models.
    """
    num_df = df[[c for c in NUMERICAL_COLS if c in df.columns]].dropna()
    corr   = num_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle only
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix (Lower Triangle)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "04_correlation_heatmap.png")


def plot_categorical_counts(df: pd.DataFrame) -> None:
    """
    Countplots for categorical columns.
    WHY: Reveals class imbalance in categories which can affect encoding decisions.
    """
    cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    fig, axes = plt.subplots(1, len(cats), figsize=(14, 5))
    if len(cats) == 1:
        axes = [axes]

    for ax, col in zip(axes, cats):
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order, palette=PALETTE, ax=ax)
        ax.set_title(f"Category Distribution: {col}", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Categorical Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "05_categorical_counts.png")


def plot_target_distribution(df: pd.DataFrame) -> None:
    """
    Show attrition (target) class balance.
    WHY: Class imbalance is one of the most common problems in real-world ML.
    """
    if TARGET_COL not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET_COL].value_counts()
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=sns.color_palette(PALETTE, 2))
    ax.set_title(f"Target Distribution: {TARGET_COL}", fontsize=13, fontweight="bold")
    ax.set_xlabel(TARGET_COL)
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha="center", fontsize=10)
    plt.tight_layout()
    _save(fig, "06_target_distribution.png")


def plot_numerical_vs_target(df: pd.DataFrame) -> None:
    """
    Boxplots of numerical features grouped by target variable.
    WHY: Tells us which features separate attrition=0 from attrition=1 best.
    """
    if TARGET_COL not in df.columns:
        return
    cols = [c for c in NUMERICAL_COLS if c in df.columns]
    fig, axes = plt.subplots(2, (len(cols) + 1) // 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.boxplot(data=df, x=TARGET_COL, y=col, palette=PALETTE, ax=axes[i])
        axes[i].set_title(f"{col} vs {TARGET_COL}", fontsize=11)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Numerical Features vs Target Variable", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "07_features_vs_target.png")


def run_full_eda(df: pd.DataFrame) -> dict:
    """Master function — run all EDA steps in sequence."""
    print("\n" + "="*60)
    print("  RUNNING FULL EDA")
    print("="*60)
    summary = initial_inspection(df)
    plot_missing_values(df)
    plot_distributions(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)
    plot_categorical_counts(df)
    plot_target_distribution(df)
    plot_numerical_vs_target(df)
    return summary
