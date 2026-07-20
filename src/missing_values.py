"""
missing_values.py
-----------------
Implements the Missing Data Decision Matrix from the PDF:

  < 5%   →  Drop rows
  5–20%  →  Median (skewed) or Group-wise conditional imputation (correlated)
  > 20%  →  KNN Imputation

Key principle: Every intervention introduces a statistical trade-off.
Choose the method that preserves the natural relationship between variables.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

from src.config import (
    MISSING_DROP_THRESHOLD,
    MISSING_STAT_THRESHOLD,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
)
from src.utils import missing_summary, skewness_report, logger


def _drop_rows_for_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Drop rows where a low-missingness column is null."""
    before = len(df)
    df = df.dropna(subset=[col])
    logger.info(f"  [drop_rows]  '{col}'  →  removed {before - len(df)} rows")
    return df


def _median_impute(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Fill missing with the global median.
    Preferred over mean for skewed distributions because the median
    is not pulled by extreme values — it's the true central tendency.
    """
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    logger.info(f"  [median]     '{col}'  →  filled with median = {median_val:.2f}")
    return df


def _group_wise_impute(df: pd.DataFrame, num_col: str, group_col: str) -> pd.DataFrame:
    """
    Fill missing values using the median within each sub-group.

    WHY: If 'salary' is missing, imputing with the global median ignores the fact
    that a 'Manager' salary is structurally different from a 'Intern' salary.
    Group-wise preserves within-group variance patterns.
    """
    df[num_col] = df.groupby(group_col)[num_col].transform(
        lambda x: x.fillna(x.median())
    )
    # Fallback: if entire group is null, use global median
    df[num_col] = df[num_col].fillna(df[num_col].median())
    logger.info(f"  [group-wise] '{num_col}' grouped by '{group_col}'")
    return df


def _knn_impute(df: pd.DataFrame, cols: list, n_neighbors: int = 5) -> pd.DataFrame:
    """
    KNN Imputation for columns with > 20% missingness.

    HOW: For each missing value, find the K most similar rows (neighbors)
    based on all other features, and take the weighted average of their values.

    WHY NOT MEAN: Mean ignores relationships between features.
    KNN captures multi-dimensional relationships — salary depends on
    department AND experience, not just one thing.

    TRADE-OFF: O(N^2) complexity — slow on very large datasets.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    df[cols] = imputer.fit_transform(df[cols])
    logger.info(f"  [KNN]        {cols}  →  imputed with {n_neighbors} neighbors")
    return df


def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Master function that applies the Decision Matrix to every column.
    Returns the cleaned DataFrame and an audit log of what was done.
    """
    print("\n" + "="*60)
    print("  MISSING VALUE TREATMENT")
    print("="*60)

    df = df.copy()
    audit = {}
    summary = missing_summary(df)

    if summary.empty:
        logger.info("No missing values found. Skipping treatment.")
        return df, audit

    print(f"\nMissing Value Summary:\n{summary}\n")

    skew_df = skewness_report(df, [c for c in NUMERICAL_COLS if c in df.columns])
    knn_candidates = []

    for col, row in summary.iterrows():
        pct = row["missing_pct"]

        # ── CASE 1: < 5% → Drop rows ──────────────────────────────────────────
        if pct < MISSING_DROP_THRESHOLD * 100:
            df = _drop_rows_for_column(df, col)
            audit[col] = {"method": "drop_rows", "missing_pct": pct}

        # ── CASE 2: 5–20% ─────────────────────────────────────────────────────
        elif pct <= MISSING_STAT_THRESHOLD * 100:
            if col in NUMERICAL_COLS:
                # Check if the column is correlated with a categorical grouper
                if "department" in df.columns and col == "salary":
                    df = _group_wise_impute(df, col, "department")
                    audit[col] = {"method": "group_wise (department)", "missing_pct": pct}
                else:
                    # Use median regardless — it's robust to skew
                    df = _median_impute(df, col)
                    skew = skew_df.loc[col, "skewness"] if col in skew_df.index else 0
                    audit[col] = {"method": "median", "missing_pct": pct, "skewness": skew}
            elif col in CATEGORICAL_COLS:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                audit[col] = {"method": "mode", "missing_pct": pct, "filled_with": mode_val}
                logger.info(f"  [mode]       '{col}'  →  filled with mode = '{mode_val}'")

        # ── CASE 3: > 20% → Queue for KNN ─────────────────────────────────────
        else:
            if col in NUMERICAL_COLS:
                knn_candidates.append(col)
                audit[col] = {"method": "KNN (queued)", "missing_pct": pct}

    # Apply KNN to all high-missingness numerical columns at once
    if knn_candidates:
        all_num = [c for c in NUMERICAL_COLS if c in df.columns]
        df = _knn_impute(df, all_num)
        for col in knn_candidates:
            audit[col]["method"] = "KNN"

    remaining = df.isnull().sum().sum()
    logger.info(f"\nMissing values remaining after treatment: {remaining}")
    return df, audit
