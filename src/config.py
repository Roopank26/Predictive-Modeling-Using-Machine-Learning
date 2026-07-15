"""Shared project configuration and column definitions."""

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "dataset"
MODELS_DIR = BASE / "models"
REPORTS_DIR = BASE / "reports"
NOTEBOOKS_DIR = BASE / "notebooks"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "loan_approved"

# Raw numeric inputs an applicant provides (ratio features are derived later).
BASE_NUMERIC_COLS = [
    "age",
    "income",
    "employment_years",
    "loan_amount",
    "loan_term",
    "credit_score",
    "existing_loans",
    "dependents",
]

# Full numeric feature list used during preprocessing (includes derived ratios).
NUMERIC_COLS = BASE_NUMERIC_COLS + ["debt_to_income", "loan_to_income"]

CATEGORICAL_COLS = [
    "gender",
    "education",
    "home_ownership",
    "loan_purpose",
    "marital_status",
]

ENGINEERED_COLS = [
    "income_per_dependent",
    "loan_per_month",
    "employment_ratio",
    "credit_tier",
    "age_group",
]

RANDOM_STATE = 42
