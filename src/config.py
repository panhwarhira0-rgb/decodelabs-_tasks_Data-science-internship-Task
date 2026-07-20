"""
config.py
---------
Central configuration for the entire pipeline.
All hardcoded values live here — never scatter magic numbers across files.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

RAW_DATA_PATH     = os.path.join(DATA_DIR, "employee_raw.csv")
CLEANED_DATA_PATH = os.path.join(OUTPUT_DIR, "cleaned_dataset.csv")

# ── Missing Value Decision Thresholds (from PDF) ───────────────────────────────
MISSING_DROP_THRESHOLD = 0.05   # < 5%  → drop rows
MISSING_STAT_THRESHOLD = 0.20   # 5–20% → median / group-wise imputation
# > 20% → KNN imputation

# ── Outlier Config ─────────────────────────────────────────────────────────────
IQR_MULTIPLIER = 1.5   # standard Tukey fence

# ── Correlation / Collinearity ─────────────────────────────────────────────────
COLLINEARITY_THRESHOLD = 0.80

# ── Dataset Columns ────────────────────────────────────────────────────────────
NUMERICAL_COLS = [
    "age", "salary", "years_experience",
    "performance_score", "hours_worked_per_week", "satisfaction_score"
]
CATEGORICAL_COLS = ["department", "education_level"]
TARGET_COL = "attrition"

# ── Plotting ───────────────────────────────────────────────────────────────────
FIGURE_DPI = 120
PALETTE    = "Set2"
