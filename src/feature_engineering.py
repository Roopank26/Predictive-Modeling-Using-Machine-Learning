"""Feature engineering utilities.

Includes:
    - create_features: row-wise meaningful feature creation (no data leakage)
    - label_encode: demonstration of Label Encoding
    - add_polynomial_features: sklearn PolynomialFeatures fit on train only
    - select_k_best / select_from_model: feature selection fit on train only
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new meaningful numeric features from existing ones (leakage-safe).

    All derived columns are computed here so that training and inference use the
    exact same feature definitions.
    """
    df = df.copy()

    # Debt-to-income and loan-to-income ratios (key credit-risk signals)
    df["debt_to_income"] = (df["existing_loans"] * 3000) / df["income"].replace(0, 1) * 100
    df["loan_to_income"] = df["loan_amount"] / df["income"].replace(0, 1)
    # Income available per dependent
    df["income_per_dependent"] = df["income"] / (df["dependents"] + 1)
    # Monthly loan burden implied by amount and term
    df["loan_per_month"] = df["loan_amount"] / df["loan_term"].replace(0, 1)
    # Share of life spent employed (experience proxy)
    df["employment_ratio"] = df["employment_years"] / df["age"].replace(0, 1)
    # Credit score band (ordinal)
    df["credit_tier"] = pd.cut(
        df["credit_score"], bins=[299, 579, 669, 739, 799, 851],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    # Age band (ordinal)
    df["age_group"] = pd.cut(
        df["age"], bins=[17, 29, 39, 49, 59, 120],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    return df


def label_encode(df: pd.DataFrame, cols: list[str]) -> Tuple[pd.DataFrame, dict]:
    """Demonstration of Label Encoding for ordinal-ish categoricals."""
    df = df.copy()
    encoders: dict = {}
    for col in cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    print(f"[label-encode] encoded {len(cols)} columns via LabelEncoder")
    return df, encoders


def add_polynomial_features(X_train, X_val, X_test, feature_names: list,
                            degree: int = 2, include_bias: bool = False):
    """Fit PolynomialFeatures on train only; transform val/test consistently."""
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias,
                              interaction_only=False)
    X_train_p = poly.fit_transform(X_train)
    X_val_p = poly.transform(X_val)
    X_test_p = poly.transform(X_test)

    names = poly.get_feature_names_out(feature_names)
    print(f"[poly] degree={degree} -> {X_train_p.shape[1]} features")
    return X_train_p, X_val_p, X_test_p, names, poly


def select_k_best(X_train, y_train, X_val, X_test, k: int = 20,
                  score_func=f_classif):
    """Select top-k features using a univariate scoring function (fit on train)."""
    selector = SelectKBest(score_func=score_func, k=min(k, X_train.shape[1]))
    X_train_s = selector.fit_transform(X_train, y_train)
    X_val_s = selector.transform(X_val)
    X_test_s = selector.transform(X_test)
    mask = selector.get_support()
    print(f"[select] kept {int(mask.sum())} of {X_train.shape[1]} features")
    return X_train_s, X_val_s, X_test_s, selector, mask


def select_from_model(estimator, X_train, y_train, X_val, X_test, threshold: str = "mean"):
    """Tree-based feature selection using a fitted estimator's importance."""
    from sklearn.feature_selection import SelectFromModel
    sfm = SelectFromModel(estimator, threshold=threshold)
    X_train_s = sfm.fit_transform(X_train, y_train)
    X_val_s = sfm.transform(X_val)
    X_test_s = sfm.transform(X_test)
    mask = sfm.get_support()
    print(f"[select-model] kept {int(mask.sum())} of {X_train.shape[1]} features")
    return X_train_s, X_val_s, X_test_s, sfm, mask
