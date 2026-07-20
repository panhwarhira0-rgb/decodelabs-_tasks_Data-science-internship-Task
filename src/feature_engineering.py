"""
feature_engineering.py
-----------------------
Create at least 3 meaningful new features from existing columns.

RULE: Never create random features. Every feature must have:
  1. A clear business reason
  2. A statistical justification
  3. Potential to improve model prediction

ALSO:
  - One-Hot Encoding for categorical columns (not Label Encoding)
  - Collinearity check and removal (threshold: |r| > 0.80)

WHY OHE not Label Encoding?
  Label Encoding assigns integers (London=1, Paris=2, Tokyo=3), creating a
  false mathematical hierarchy. OHE maps each category to its own binary axis
  so they're equidistant in coordinate space (as shown in the PDF diagram).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.config import (
    NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL,
    COLLINEARITY_THRESHOLD, PLOTS_DIR, FIGURE_DPI
)
from src.utils import logger


# ── Feature 1: Salary Per Year of Experience ──────────────────────────────────
def add_salary_per_experience(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business meaning: Employees underpaid relative to their experience
    are more likely to leave (higher attrition risk).

    Mathematical logic: salary / years_experience = compensation efficiency.
    A senior employee earning the same as a junior is a red flag the model
    should detect. We add 1 to avoid division by zero for new hires.
    """
    df["salary_per_experience"] = df["salary"] / (df["years_experience"] + 1)
    logger.info("  [FE] Added: salary_per_experience")
    return df


# ── Feature 2: Work-Life Imbalance Score ──────────────────────────────────────
def add_workload_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business meaning: Overworked employees with low satisfaction scores
    are a very strong predictor of attrition.

    Mathematical logic: We combine two features — hours worked (stress proxy)
    and satisfaction score (wellbeing proxy) — into a single risk signal.
    High hours + low satisfaction = high workload score = higher risk.

    We invert satisfaction (10 - satisfaction) so high numbers mean dissatisfied,
    then multiply by normalized hours to get a compound risk score.
    """
    df["workload_score"] = (
        (df["hours_worked_per_week"] / df["hours_worked_per_week"].max()) *
        (10 - df["satisfaction_score"])
    )
    logger.info("  [FE] Added: workload_score")
    return df


# ── Feature 3: Experience Ratio ────────────────────────────────────────────────
def add_experience_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business meaning: How much of someone's potential career have they spent
    at this company? Younger employees with many years experience might be
    more specialized and harder to replace.

    Mathematical logic: years_experience / age gives a ratio between 0 and 1.
    We add 1 to age to avoid edge cases.
    """
    df["experience_ratio"] = df["years_experience"] / (df["age"] + 1)
    logger.info("  [FE] Added: experience_ratio")
    return df


# ── Feature 4 (Bonus): Performance-Satisfaction Index ─────────────────────────
def add_perf_satisfaction_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business meaning: A high performer who is also highly satisfied is
    a valuable, stable employee. Low performer + dissatisfied = flight risk.

    This interaction feature captures both dimensions simultaneously.
    """
    df["perf_satisfaction_index"] = df["performance_score"] * df["satisfaction_score"]
    logger.info("  [FE] Added: perf_satisfaction_index")
    return df


# ── One-Hot Encoding ───────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply One-Hot Encoding to all categorical columns.
    drop_first=True removes one dummy variable per group to avoid the
    'dummy variable trap' (perfect multicollinearity).
    """
    cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    if not cats:
        return df
    df = pd.get_dummies(df, columns=cats, drop_first=True, dtype=int)
    logger.info(f"  [OHE] Encoded: {cats}  →  new shape: {df.shape}")
    return df


# ── Collinearity Eradication ───────────────────────────────────────────────────
def remove_collinear_features(df: pd.DataFrame, target: str = TARGET_COL) -> tuple[pd.DataFrame, list]:
    """
    Implements the 4-step Collinearity Eradication Algorithm from the PDF:
      1. Build absolute correlation matrix
      2. Isolate upper triangle (avoid duplicate pairs)
      3. Identify pairs with |r| > threshold
      4. Drop the feature with WEAKER correlation to the target

    WHY: Multicollinearity makes the feature matrix singular (non-invertible),
    causing OLS coefficients to become violently unstable.
    """
    num_df  = df.select_dtypes(include=[np.number])
    abs_corr = num_df.corr().abs()
    upper    = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        correlated_pairs = upper.index[upper[col] > COLLINEARITY_THRESHOLD].tolist()
        for paired_col in correlated_pairs:
            if target in df.columns:
                # Keep the feature more correlated with the target
                corr_col    = abs(df[col].corr(df[target]))
                corr_paired = abs(df[paired_col].corr(df[target]))
                weaker = col if corr_col < corr_paired else paired_col
            else:
                weaker = col
            to_drop.add(weaker)
            logger.info(
                f"  [collinearity] '{col}' & '{paired_col}' → dropping '{weaker}'"
            )

    df = df.drop(columns=list(to_drop), errors="ignore")
    return df, list(to_drop)


def plot_engineered_features(df: pd.DataFrame) -> None:
    """Visualize the distribution of newly engineered features."""
    new_features = [
        "salary_per_experience", "workload_score",
        "experience_ratio", "perf_satisfaction_index"
    ]
    existing = [f for f in new_features if f in df.columns]
    if not existing:
        return

    fig, axes = plt.subplots(1, len(existing), figsize=(5 * len(existing), 4))
    if len(existing) == 1:
        axes = [axes]

    for ax, feat in zip(axes, existing):
        sns.histplot(df[feat], kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution: {feat}", fontsize=10)
        ax.set_xlabel(feat)

    fig.suptitle("Engineered Feature Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "09_engineered_features.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot saved]  09_engineered_features.png")


def run_feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Master function: run all FE steps and return enriched DataFrame + report."""
    print("\n" + "="*60)
    print("  FEATURE ENGINEERING")
    print("="*60)

    original_cols = list(df.columns)
    df = add_salary_per_experience(df)
    df = add_workload_score(df)
    df = add_experience_ratio(df)
    df = add_perf_satisfaction_index(df)
    df = encode_categoricals(df)
    df, dropped = remove_collinear_features(df)
    plot_engineered_features(df)

    report = {
        "original_columns": original_cols,
        "features_added": [
            "salary_per_experience", "workload_score",
            "experience_ratio", "perf_satisfaction_index"
        ],
        "dropped_collinear": dropped,
        "final_shape": list(df.shape),
        "final_columns": list(df.columns),
    }
    logger.info(f"  Final dataset shape: {df.shape}")
    return df, report
