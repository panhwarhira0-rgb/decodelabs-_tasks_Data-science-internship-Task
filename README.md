# Project 1: Advanced EDA & Feature Engineering

**DecodeLabs Industrial Training Kit — Batch 2026**

---

## Project Overview

Transform a raw, messy HR employee dataset into a mathematically clean, ML-ready feature store. This project covers the complete data preprocessing pipeline used in production Data Science systems — not just running algorithms, but understanding every statistical decision made along the way.

**Business Problem:** Employee attrition costs companies 50–200% of an employee's annual salary. Before building a predictive model, we need clean, high-signal features.

---

## Project Structure

```
Project1/
│
├── data/
│   ├── employee_raw.csv          # Raw dataset (with missing values & outliers)
│   └── generate_dataset.py       # Script to regenerate the dataset
│
├── src/
│   ├── config.py                 # All paths and hyperparameters
│   ├── utils.py                  # Shared helper functions
│   ├── eda.py                    # Exploratory Data Analysis
│   ├── missing_values.py         # Missing data treatment (Decision Matrix)
│   ├── outliers.py               # IQR detection + Winsorization
│   ├── feature_engineering.py    # Feature creation + OHE + collinearity
│   └── data_contracts.py         # Pandera schema validation
│
├── outputs/
│   ├── cleaned_dataset.csv       # Final cleaned dataset
│   └── plots/                    # All EDA and post-treatment visualizations
│
├── reports/
│   ├── eda_report.json
│   ├── outlier_report.json
│   └── feature_engineering_report.json
│
├── main.py                       # Pipeline entry point
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone <your-repo-url>
cd Project1
pip install -r requirements.txt
```

---

## How to Run

```bash
# Step 1: Generate the dataset
python data/generate_dataset.py

# Step 2: Run the full pipeline
python main.py
```

---

## Pipeline Architecture (IPO)

| Stage  | Module               | Actions                                          |
|--------|----------------------|--------------------------------------------------|
| INPUT  | missing_values.py    | Drop (<5%), Median/Group-wise (5-20%), KNN (>20%)|
| INPUT  | outliers.py          | IQR detection + numpy.clip() Winsorization       |
| PROCESS| feature_engineering.py | 4 new features + OHE + collinearity eradication|
| OUTPUT | data_contracts.py    | Pandera runtime schema validation                |

---

## Features Engineered

| Feature                  | Business Logic                                              |
|--------------------------|-------------------------------------------------------------|
| `salary_per_experience`  | Compensation fairness index — underpaid seniors churn more  |
| `workload_score`         | Compound stress signal: high hours × low satisfaction       |
| `experience_ratio`       | Career density — experience relative to age                 |
| `perf_satisfaction_index`| Retention stability: high performance × high satisfaction   |

---

## Key Statistical Concepts Used

- **IQR (Interquartile Range):** Non-parametric outlier detection robust to skewed distributions
- **Winsorization:** Capping at fence boundaries to preserve row count
- **KNN Imputation:** Multi-dimensional estimation for high-missingness columns
- **One-Hot Encoding:** Orthogonal coordinate mapping for nominal categories
- **Multicollinearity Eradication:** Pearson correlation threshold at |r| > 0.80
- **Pandera Data Contracts:** Runtime schema enforcement for production pipelines

---

## Expected Outputs

- `outputs/cleaned_dataset.csv` — 970 rows × 18 features, zero nulls
- `outputs/plots/` — 9 visualizations (distributions, heatmap, boxplots before/after, etc.)
- `reports/` — JSON audit logs for every transformation

---

## Learning Outcomes

By completing this project you will understand:
- How to make principled imputation decisions (not just `fillna(mean)`)
- Why IQR is preferred over Z-Score for real-world business data
- The mathematical danger of multicollinearity in linear models
- How feature engineering creates business-meaningful signals
- Why production ML systems need runtime data contracts
