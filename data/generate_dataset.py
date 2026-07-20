"""
generate_dataset.py
-------------------
Generates a realistic synthetic Employee HR dataset with:
  - Intentional missing values (at 3 different missingness levels)
  - Injected outliers
  - Both numerical and categorical columns

Run this once to create employee_raw.csv in the data/ folder.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1000

departments   = ["Engineering", "Sales", "HR", "Marketing", "Finance", "Operations"]
edu_levels    = ["High School", "Bachelor", "Master", "PhD"]

data = {
    "employee_id":           range(1001, 1001 + N),
    "age":                   np.random.randint(22, 60, N).astype(float),
    "department":            np.random.choice(departments, N),
    "salary":                np.random.normal(60000, 15000, N).clip(25000, 150000),
    "years_experience":      np.random.randint(0, 35, N).astype(float),
    "performance_score":     np.round(np.random.uniform(1, 10, N), 1),
    "hours_worked_per_week": np.random.normal(42, 8, N).clip(20, 80),
    "education_level":       np.random.choice(edu_levels, N, p=[0.1, 0.5, 0.3, 0.1]),
    "satisfaction_score":    np.round(np.random.uniform(1, 10, N), 1),
    "attrition":             np.random.choice([0, 1], N, p=[0.75, 0.25]),
}

df = pd.DataFrame(data)

# ── Inject outliers ────────────────────────────────────────────────────────────
outlier_idx = np.random.choice(N, 25, replace=False)
df.loc[outlier_idx[:10], "salary"]                = np.random.uniform(300000, 500000, 10)
df.loc[outlier_idx[10:18], "hours_worked_per_week"] = np.random.uniform(90, 110, 8)
df.loc[outlier_idx[18:], "age"]                  = np.random.uniform(80, 100, 7)

# ── Inject missing values (3 different levels) ─────────────────────────────────
# < 5%  → performance_score (3%)
perf_missing = np.random.choice(N, int(N * 0.03), replace=False)
df.loc[perf_missing, "performance_score"] = np.nan

# 5–20% → satisfaction_score (10%), hours_worked_per_week (8%)
sat_missing = np.random.choice(N, int(N * 0.10), replace=False)
df.loc[sat_missing, "satisfaction_score"] = np.nan

hrs_missing = np.random.choice(N, int(N * 0.08), replace=False)
df.loc[hrs_missing, "hours_worked_per_week"] = np.nan

# 5–20% with group structure → salary (12%, group-wise imputation makes sense)
sal_missing = np.random.choice(N, int(N * 0.12), replace=False)
df.loc[sal_missing, "salary"] = np.nan

# > 20% → years_experience (22%, KNN needed)
exp_missing = np.random.choice(N, int(N * 0.22), replace=False)
df.loc[exp_missing, "years_experience"] = np.nan

# 5% → education_level (categorical, mode imputation)
edu_missing = np.random.choice(N, int(N * 0.05), replace=False)
df.loc[edu_missing, "education_level"] = np.nan

out_path = os.path.join(os.path.dirname(__file__), "employee_raw.csv")
df.to_csv(out_path, index=False)
print(f"Dataset generated: {df.shape[0]} rows × {df.shape[1]} cols  →  {out_path}")
print(f"Missing value summary:\n{df.isnull().sum()}")
