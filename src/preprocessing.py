"""Data preprocessing utilities.

Responsibilities:
    - load raw train/test CSVs
    - clean data (drop duplicates, impute missing, remove outliers)
    - encode categorical variables (one-hot)
    - scale numeric features with StandardScaler (fitted on train only)
    - split into train/validation/test arrays

Design notes (correctness / no data leakage):
    - Imputation statistics (median for numeric, mode for categorical) are computed
      from the TRAINING set only, then applied to validation/test.
    - Outliers are removed from the TRAINING set only. The held-out test set keeps its
      natural distribution so evaluation reflects real-world performance.
    - The StandardScaler is fitted on the training split only and reused for val/test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config as cfg
from feature_engineering import create_features

DATA_DIR = cfg.DATA_DIR
MODELS_DIR = cfg.MODELS_DIR
TARGET = cfg.TARGET
NUMERIC_COLS = cfg.NUMERIC_COLS
CATEGORICAL_COLS = cfg.CATEGORICAL_COLS
IMPUTE_STATS_FILE = MODELS_DIR / "imputation_stats.json"


def load_data(train_path: Path = DATA_DIR / "train.csv",
              test_path: Path = DATA_DIR / "test.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw training and test CSVs."""
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training data at {train_path}. Run generate_dataset.py first.")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test data at {test_path}. Run generate_dataset.py first.")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[clean] removed {before - len(df)} duplicate rows (kept {len(df)})")
    return df


def compute_imputation_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Compute imputation values from a DataFrame (intended to be the TRAIN set).

    Numeric columns -> median; categorical columns -> most frequent mode.
    """
    stats: Dict[str, float] = {}
    for col in NUMERIC_COLS:
        if col in df.columns:
            stats[col] = float(df[col].median())
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].notna().any():
            stats[col] = str(df[col].mode(dropna=True).iloc[0])
    return stats


def apply_imputation(df: pd.DataFrame, stats: Dict[str, float]) -> pd.DataFrame:
    """Fill missing values using precomputed statistics (train-fitted)."""
    df = df.copy()
    for col, value in stats.items():
        if col in df.columns and df[col].isna().any():
            n = int(df[col].isna().sum())
            df[col] = df[col].fillna(value)
            kind = "numeric" if isinstance(value, (int, float)) else "categorical"
            print(f"[impute] {kind} '{col}': filled {n} missing with {value}")
    return df


def save_imputation_stats(stats: Dict[str, float]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    IMPUTE_STATS_FILE.write_text(json.dumps(stats))
    print(f"[impute] saved stats -> {IMPUTE_STATS_FILE}")


def load_imputation_stats() -> Dict[str, float]:
    if not IMPUTE_STATS_FILE.exists():
        raise FileNotFoundError(
            f"Imputation stats not found at {IMPUTE_STATS_FILE}. "
            "Train the pipeline (src/pipeline.py) before running predictions."
        )
    return json.loads(IMPUTE_STATS_FILE.read_text())


def remove_outliers(df: pd.DataFrame, method: str = "iqr", factor: float = 1.5) -> pd.DataFrame:
    """Remove rows whose numeric features fall outside the IQR fences."""
    df = df.copy()
    before = len(df)
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        df = df[(df[col] >= lo) & (df[col] <= hi)]
    df = df.reset_index(drop=True)
    print(f"[outliers] removed {before - len(df)} rows via IQR (kept {len(df)})")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns (drop_first=False for completeness)."""
    df = df.copy()
    cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cats, drop_first=False)
    print(f"[encode] one-hot encoded {len(cats)} categorical columns -> {df.shape[1]} total columns")
    return df


def scale_features(X: pd.DataFrame, scaler: StandardScaler | None = None,
                   save: bool = True) -> Tuple[np.ndarray, StandardScaler]:
    """Scale features with StandardScaler. Fits only when no scaler is supplied."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X)
        print("[scale] fitted StandardScaler on training data")
    Xs = scaler.transform(X)
    if save:
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
        print(f"[scale] saved scaler -> {MODELS_DIR / 'scaler.pkl'}")
    return Xs, scaler


def prepare_pipeline(train: pd.DataFrame, test: pd.DataFrame, target: str = TARGET,
                     exclude_cols: list | None = None,
                     test_size: float = 0.2, random_state: int = 42,
                     stratify: object = "auto",
                     save_scaler: bool = True) -> Tuple:
    """Full preprocessing pipeline returning scaled train/val/test splits + feature list.

    Args:
        train/test: raw DataFrames.
        target: column to predict.
        exclude_cols: columns to drop from features (targets / leakage-derived).
        test_size: validation split fraction.
        random_state: seed for reproducible splitting.
        stratify: "auto" stratifies when the target is low-cardinality, else None.
        save_scaler: whether to persist the fitted scaler (only for the primary task).

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler)
    """
    # --- Training set: dedup -> fit imputation -> impute -> remove outliers ---
    train_dedup = drop_duplicates(train)
    stats = compute_imputation_stats(train_dedup)
    save_imputation_stats(stats)
    train_imp = apply_imputation(train_dedup, stats)
    train_clean = remove_outliers(train_imp)

    # --- Test set: impute with TRAIN statistics only (no dedup, no outlier removal) ---
    test_clean = apply_imputation(test, stats)

    train_fe = create_features(train_clean)
    test_fe = create_features(test_clean)

    train_enc = encode_categoricals(train_fe)
    test_enc = encode_categoricals(test_fe)

    exclude_cols = exclude_cols or []
    feature_cols = [c for c in train_enc.columns
                    if c != target and c not in exclude_cols]
    # Align test columns to the training schema (unseen -> 0, missing -> 0)
    test_enc = test_enc.reindex(columns=feature_cols + [target], fill_value=0)

    X = train_enc[feature_cols]
    y = train_enc[target]
    X_test = test_enc[feature_cols]
    y_test = test_enc[target]

    if stratify == "auto":
        stratify = y if y.nunique() <= 20 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    X_train_s, scaler = scale_features(X_train, save=save_scaler)
    X_val_s, _ = scale_features(X_val, scaler=scaler, save=False)
    X_test_s, _ = scale_features(X_test, scaler=scaler, save=False)

    return (X_train_s, X_val_s, X_test_s,
            y_train.values, y_val.values, y_test.values,
            feature_cols, scaler)
