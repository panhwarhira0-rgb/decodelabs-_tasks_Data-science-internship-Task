"""
outliers.py
-----------
Outlier detection using IQR (Interquartile Range) method.
Treatment via Winsorization (capping) — NOT deletion.

WHY IQR over Z-Score?
  Z-Score assumes a normal distribution. Real-world data (salaries, hours worked)
  is often skewed. IQR is non-parametric — it makes no distribution assumption,
  making it more robust for this dataset.

WHY WINSORIZATION over DELETION?
  Deleting outlier rows destroys adjacent feature values (other columns in that row).
  Capping at the boundary preserves row count and sequential integrity,
  as shown in the PDF's "Winsorization vs Deletion" diagram.
  We use numpy.clip() to cap values at the IQR fence.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.config import IQR_MULTIPLIER, NUMERICAL_COLS, PLOTS_DIR, FIGURE_DPI, PALETTE
from src.utils import logger


def detect_outliers_iqr(df: pd.DataFrame, col: str) -> dict:
    """
    Calculate IQR boundaries and count outliers for a single column.

    Returns a dict with Q1, Q3, IQR, lower_bound, upper_bound, outlier_count.
    """
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - IQR_MULTIPLIER * IQR
    upper = Q3 + IQR_MULTIPLIER * IQR

    outlier_mask  = (df[col] < lower) | (df[col] > upper)
    outlier_count = outlier_mask.sum()

    return {
        "Q1": Q1, "Q3": Q3, "IQR": IQR,
        "lower_bound": lower, "upper_bound": upper,
        "outlier_count": int(outlier_count),
        "outlier_pct": round(outlier_count / len(df) * 100, 2),
    }


def winsorize_column(df: pd.DataFrame, col: str, lower: float, upper: float) -> pd.DataFrame:
    """
    Cap values at IQR boundaries using numpy.clip().
    Values below lower → set to lower.
    Values above upper → set to upper.
    """
    df[col] = np.clip(df[col], lower, upper)
    return df


def plot_boxplots_after(df: pd.DataFrame) -> None:
    """Boxplots AFTER treatment to visually confirm outliers are neutralized."""
    cols = [c for c in NUMERICAL_COLS if c in df.columns]
    fig, axes = plt.subplots(2, (len(cols) + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.boxplot(y=df[col], ax=axes[i], color=sns.color_palette(PALETTE)[i % 6])
        axes[i].set_title(f"Boxplot (After): {col}", fontsize=11)
        axes[i].set_ylabel(col)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Boxplots — After Outlier Treatment (Winsorization)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "08_boxplots_after.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot saved]  08_boxplots_after.png")


def handle_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Master function: detect and treat outliers for all numerical columns.
    Returns the cleaned DataFrame and an audit report.
    """
    print("\n" + "="*60)
    print("  OUTLIER DETECTION & TREATMENT (IQR + WINSORIZATION)")
    print("="*60)

    df     = df.copy()
    report = {}
    cols   = [c for c in NUMERICAL_COLS if c in df.columns]

    for col in cols:
        stats = detect_outliers_iqr(df, col)
        report[col] = stats

        if stats["outlier_count"] > 0:
            logger.info(
                f"  '{col}'  →  {stats['outlier_count']} outliers ({stats['outlier_pct']}%)  "
                f"| bounds: [{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]"
            )
            df = winsorize_column(df, col, stats["lower_bound"], stats["upper_bound"])
        else:
            logger.info(f"  '{col}'  →  no outliers detected")

    plot_boxplots_after(df)
    return df, report
