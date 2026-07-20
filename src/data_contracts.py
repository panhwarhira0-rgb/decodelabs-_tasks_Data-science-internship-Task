"""
data_contracts.py
-----------------
Pandera schema validation — the final gate before the dataset is saved.

WHY DATA CONTRACTS?
  In production, data arrives from multiple sources. Silent data corruption
  (wrong dtypes, values outside expected ranges) can poison an ML model
  without raising any Python error. Pandera enforces explicit structural
  contracts at runtime, catching issues before they propagate downstream.

  Using lazy=True collects ALL violations into one diagnostic report
  instead of crashing on the first error.
"""

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from src.utils import logger


def build_schema() -> DataFrameSchema:
    """
    Define the expected structure of the cleaned dataset.
    These contracts are derived from business rules and domain knowledge.
    """
    schema = DataFrameSchema(
        {
            "age": Column(
                float,
                checks=[Check.greater_than_or_equal_to(18), Check.less_than_or_equal_to(70)],
                nullable=False,
                description="Employee age must be between 18 and 70.",
            ),
            "salary": Column(
                float,
                checks=Check.greater_than(0),
                nullable=False,
                description="Salary must be a positive number.",
            ),
            "years_experience": Column(
                float,
                checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(50)],
                nullable=False,
            ),
            "performance_score": Column(
                float,
                checks=[Check.in_range(1, 10)],
                nullable=False,
            ),
            "hours_worked_per_week": Column(
                float,
                checks=[Check.in_range(0, 100)],
                nullable=False,
            ),
            "satisfaction_score": Column(
                float,
                checks=[Check.in_range(1, 10)],
                nullable=False,
            ),
            "salary_per_experience": Column(float, nullable=False),
            "workload_score":        Column(float, nullable=False),
            "experience_ratio":      Column(float, nullable=False),
            "perf_satisfaction_index": Column(float, nullable=False),
        },
        coerce=True,   # attempt dtype coercion before checking
    )
    return schema


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Run Pandera validation with lazy=True.
    Returns True if data passes all checks, False otherwise.
    Failures are logged but do NOT crash the pipeline.
    """
    print("\n" + "="*60)
    print("  DATA CONTRACT VALIDATION (Pandera)")
    print("="*60)

    schema = build_schema()

    try:
        schema.validate(df, lazy=True)
        logger.info("  All data contract checks PASSED.")
        return True
    except pa.errors.SchemaErrors as exc:
        logger.warning(f"  Data contract FAILURES detected:\n{exc.failure_cases}")
        return False
