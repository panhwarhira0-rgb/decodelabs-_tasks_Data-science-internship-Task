"""
utils.py
--------
Reusable helper functions shared across the pipeline.
Think of this as the project's toolbox.
"""

import os
import json
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame with basic logging."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset  →  {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved  →  {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")


def save_report(report: dict, path: str) -> None:
    """Persist a Python dict as a JSON report file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=4, default=str)
    logger.info(f"Report saved  →  {path}")


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a ranked summary of missing values per column.

    Returns
    -------
    pd.DataFrame with columns: [missing_count, missing_pct, recommended_strategy]
    """
    missing_count = df.isnull().sum()
    missing_pct   = (missing_count / len(df)) * 100

    def strategy(pct):
        if pct == 0:
            return "none"
        elif pct < 5:
            return "drop_rows"
        elif pct <= 20:
            return "median / group-wise"
        else:
            return "KNN imputation"

    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct":   missing_pct.round(2),
        "recommended_strategy": missing_pct.map(strategy),
    })
    return summary[summary["missing_count"] > 0].sort_values("missing_pct", ascending=False)


def skewness_report(df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    """Return skewness values for numerical columns to guide imputation choice."""
    skew_vals = df[numerical_cols].skew().round(3)
    return pd.DataFrame({
        "skewness": skew_vals,
        "distribution": skew_vals.map(
            lambda s: "symmetric" if abs(s) < 0.5 else "skewed"
        )
    })
