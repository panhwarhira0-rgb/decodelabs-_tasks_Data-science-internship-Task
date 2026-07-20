"""
main.py
-------
The orchestrator. This is the single entry point for the entire pipeline.
Run it with: python main.py

Pipeline flow (IPO Architecture from PDF):
  INPUT  → load + EDA
  PROCESS → missing values + outliers + feature engineering
  OUTPUT  → validation + save cleaned dataset + reports
"""

import os
import sys

# Make src importable from root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config      import RAW_DATA_PATH, CLEANED_DATA_PATH, REPORTS_DIR
from src.utils       import load_csv, save_csv, save_report, missing_summary, logger
from src.eda         import run_full_eda
from src.missing_values   import handle_missing_values
from src.outliers         import handle_outliers
from src.feature_engineering import run_feature_engineering
from src.data_contracts   import validate_dataset


def main():
    print("\n" + "█"*60)
    print("  PROJECT 1: ADVANCED EDA & FEATURE ENGINEERING")
    print("  Powered by DecodeLabs  |  Batch 2026")
    print("█"*60)

    # ── STEP 1: Load ───────────────────────────────────────────────────────────
    df_raw = load_csv(RAW_DATA_PATH)

    # ── STEP 2: EDA on raw data ────────────────────────────────────────────────
    eda_summary = run_full_eda(df_raw)

    # ── STEP 3: Missing Value Treatment ───────────────────────────────────────
    df_imputed, missing_report = handle_missing_values(df_raw)

    # ── STEP 4: Outlier Treatment ──────────────────────────────────────────────
    df_clean, outlier_report = handle_outliers(df_imputed)

    # ── STEP 5: Feature Engineering ───────────────────────────────────────────
    df_final, fe_report = run_feature_engineering(df_clean)

    # ── STEP 6: Data Contract Validation ──────────────────────────────────────
    passed = validate_dataset(df_final)

    # ── STEP 7: Save Outputs ──────────────────────────────────────────────────
    save_csv(df_final, CLEANED_DATA_PATH)

    save_report(
        {"eda": eda_summary, "missing_values": missing_report},
        os.path.join(REPORTS_DIR, "eda_report.json")
    )
    save_report(
        {"outliers": outlier_report},
        os.path.join(REPORTS_DIR, "outlier_report.json")
    )
    save_report(
        {"feature_engineering": fe_report, "validation_passed": passed},
        os.path.join(REPORTS_DIR, "feature_engineering_report.json")
    )

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print(f"  Cleaned dataset  →  {CLEANED_DATA_PATH}")
    print(f"  Reports          →  {REPORTS_DIR}/")
    print(f"  Plots            →  outputs/plots/")
    print(f"  Validation       →  {'PASSED' if passed else 'WARNINGS — see report'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
