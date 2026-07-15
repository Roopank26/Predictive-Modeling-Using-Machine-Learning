"""Prediction on new, raw records using the saved model + scaler + feature schema.

Replicates the training feature pipeline (clean -> engineer -> one-hot -> scale)
using the TRAIN-fitted imputation statistics and scaler, then outputs the approval
probability and label. No rows are dropped during inference.
"""

from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import config as cfg
from feature_engineering import create_features
from preprocessing import (apply_imputation, encode_categoricals,
                           load_imputation_stats)
from train_model import load_model

MODELS_DIR = cfg.MODELS_DIR
FEATURES_FILE = MODELS_DIR / "features.json"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
REQUIRED_COLUMNS = cfg.BASE_NUMERIC_COLS + cfg.CATEGORICAL_COLS


def _load_artifacts() -> Tuple[object, object, list]:
    """Load the persisted model, scaler, and feature schema with clear errors."""
    missing = [p for p in (FEATURES_FILE, SCALER_FILE, MODELS_DIR / "trained_model.pkl")
               if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing artifact(s): " + ", ".join(str(m) for m in missing) +
            ". Train the pipeline first with `python src/pipeline.py`."
        )
    model = load_model("trained_model.pkl")
    scaler = joblib.load(SCALER_FILE)
    feature_cols = json.loads(FEATURES_FILE.read_text())
    return model, scaler, feature_cols


def predict_new(raw_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Predict loan approval for one or more raw applicant records.

    Args:
        raw_df: DataFrame with the raw input columns (see REQUIRED_COLUMNS).
        threshold: probability cutoff for the positive ('Approved') class.

    Returns:
        DataFrame with `approval_probability` and `prediction` columns.
    """
    model, scaler, feature_cols = _load_artifacts()
    stats = load_imputation_stats()

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Input is missing required column(s): {missing_cols}")

    df = apply_imputation(raw_df.copy(), stats)
    df = create_features(df)
    df = encode_categoricals(df)
    # Align to the training schema: unseen categories -> 0, absent features -> 0
    df = df.reindex(columns=feature_cols, fill_value=0)

    Xs = scaler.transform(df[feature_cols])
    proba = model.predict_proba(Xs)[:, 1]
    pred_labels = np.where(proba >= threshold, "Approved", "Rejected")
    return pd.DataFrame({
        "approval_probability": np.round(proba, 4),
        "prediction": pred_labels,
    })


def demo() -> None:
    sample = pd.DataFrame([{
        "age": 35, "income": 85000, "employment_years": 10, "loan_amount": 15000,
        "loan_term": 36, "credit_score": 720, "existing_loans": 1, "dependents": 2,
        "gender": "Male", "education": "Bachelor", "home_ownership": "Own",
        "loan_purpose": "Home", "marital_status": "Married",
    }])
    print(predict_new(sample).to_string(index=False))


if __name__ == "__main__":
    demo()
