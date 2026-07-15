"""Generate a realistic, self-contained Loan Approval dataset.

Produces:
    dataset/train.csv  (with target column `loan_approved`)
    dataset/test.csv   (held-out set, also with target for evaluation)

The dataset intentionally includes:
    - mixed numeric and categorical features
    - missing values (NaN) to demonstrate imputation
    - duplicate records to demonstrate de-duplication
    - outliers to demonstrate outlier handling
    - meaningful signal between features and the target

Note: the `debt_to_income` and `loan_to_income` ratio features are intentionally
NOT stored here. They are derived in `feature_engineering.create_features` so that
training and inference share one definition.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

N_TRAIN = 4000
N_TEST = 1000

NUMERIC_BASE = {
    "age": (25, 70, 12.0),
    "income": (20000, 220000, 35000.0),
    "employment_years": (0, 40, 8.0),
    "loan_amount": (1000, 50000, 8000.0),
    "loan_term": (6, 360, 80.0),
    "credit_score": (300, 850, 90.0),
    "existing_loans": (0, 8, 1.5),
    "dependents": (0, 5, 1.2),
}

CATEGORICALS = {
    "gender": ["Male", "Female"],
    "education": ["High School", "Bachelor", "Master", "PhD"],
    "home_ownership": ["Rent", "Mortgage", "Own"],
    "loan_purpose": ["Home", "Car", "Education", "Personal", "Business"],
    "marital_status": ["Single", "Married", "Divorced"],
}


def _make_block(n: int) -> pd.DataFrame:
    data = {}
    for col, (lo, hi, sd) in NUMERIC_BASE.items():
        center = (lo + hi) / 2
        vals = RNG.normal(center, sd, size=n)
        vals = np.clip(vals, lo, hi)
        data[col] = np.round(vals, 1)

    for col, opts in CATEGORICALS.items():
        data[col] = RNG.choice(opts, size=n)

    df = pd.DataFrame(data)

    # Latent ratio features (kept local; mirrored in feature_engineering.create_features)
    debt_to_income = np.round(
        (df["existing_loans"] * 3000) / df["income"].replace(0, 1) * 100, 2)
    loan_to_income = np.round(df["loan_amount"] / df["income"].replace(0, 1), 3)

    # Education / home ownership numeric boosts (latent)
    edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    own_map = {"Rent": 0, "Mortgage": 1, "Own": 2}
    edu = df["education"].map(edu_map)
    own = df["home_ownership"].map(own_map)

    # Logistic latent score -> probability of approval
    score = (
        0.010 * (df["credit_score"] - 600)
        + 0.000004 * (df["income"] - 60000)
        + 0.05 * df["employment_years"]
        - 1.2 * loan_to_income
        - 0.05 * debt_to_income
        + 0.15 * edu
        + 0.10 * own
        - 0.10 * df["existing_loans"]
        + 0.02 * (df["age"] - 40)
    )
    prob = 1 / (1 + np.exp(-score))
    df["loan_approved"] = (RNG.random(n) < prob).astype(int)

    # Continuous regression target: loan interest rate (has real signal)
    noise = RNG.normal(0, 0.4, size=n)
    df["interest_rate"] = (
        3.0
        + (700 - df["credit_score"]) / 100 * 1.6
        + debt_to_income * 0.02
        + (loan_to_income - 0.3).clip(lower=0) * 4.0
        - 0.3 * own
        + noise
    ).clip(2.0, 25.0)
    df["interest_rate"] = np.round(df["interest_rate"], 3)

    return df


def _inject_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_cols = ["income", "credit_score", "employment_years", "loan_amount", "education"]
    for col in missing_cols:
        mask = RNG.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan
    return df


def _inject_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    idx = RNG.choice(n, size=int(0.03 * n), replace=False)
    df.loc[idx, "income"] = df.loc[idx, "income"] * RNG.uniform(3, 6, size=len(idx))
    df.loc[idx, "loan_amount"] = df.loc[idx, "loan_amount"] * RNG.uniform(3, 8, size=len(idx))
    df.loc[idx, "credit_score"] = RNG.uniform(300, 850, size=len(idx))
    return df


def _inject_duplicates(df: pd.DataFrame, frac: float = 0.04) -> pd.DataFrame:
    dup = df.sample(frac=frac, random_state=7)
    return pd.concat([df, dup], ignore_index=True)


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = _make_block(N_TRAIN)
    test = _make_block(N_TEST)

    train = _inject_missing(train)
    test = _inject_missing(test)

    train = _inject_outliers(train)
    test = _inject_outliers(test)

    train = _inject_duplicates(train)

    # Deterministic column order (ratio features are derived later in create_features)
    cols = (
        list(NUMERIC_BASE.keys())
        + list(CATEGORICALS.keys())
        + ["interest_rate", "loan_approved"]
    )
    train = train[cols]
    test = test[cols]

    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    print(f"Wrote {len(train)} rows to {out_dir / 'train.csv'}")
    print(f"Wrote {len(test)} rows to {out_dir / 'test.csv'}")
    print(f"Train approval rate: {train['loan_approved'].mean():.3f}")
    print(f"Test  approval rate: {test['loan_approved'].mean():.3f}")


if __name__ == "__main__":
    main()
