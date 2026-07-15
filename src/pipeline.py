"""End-to-end pipeline orchestrator.

Run with:  python src/pipeline.py

Performs: load -> clean -> feature engineering -> encode -> scale -> split ->
train & tune multiple models -> select best -> evaluate -> visualize ->
train a secondary regression demo -> save artifacts.
"""

from __future__ import annotations

import json
import matplotlib
matplotlib.use("Agg")  # headless: no GUI backend needed when run as a script

import config as cfg
from evaluate import evaluate_classifier, evaluate_regression, regression_metrics
from feature_engineering import select_k_best
from preprocessing import load_data, prepare_pipeline
from train_model import (save_model, select_best_classifier,
                         train_classification, train_regression)

MODELS_DIR = cfg.MODELS_DIR


def run_full_pipeline() -> dict:
    print("\n" + "=" * 70)
    print("1. LOAD DATA")
    print("=" * 70)
    train, test = load_data()
    print(f"train shape={train.shape}, test shape={test.shape}")

    print("\n" + "=" * 70)
    print("2. PREPROCESS + FEATURE ENGINEERING + SPLIT")
    print("=" * 70)
    (X_train, X_val, X_test,
     y_train, y_val, y_test, feature_cols, scaler) = prepare_pipeline(
        train, test, test_size=0.2, random_state=cfg.RANDOM_STATE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "features.json").write_text(json.dumps(feature_cols))
    print(f"feature count = {len(feature_cols)}")

    print("\n" + "=" * 70)
    print("3. TRAIN + TUNE CLASSIFIERS (GridSearchCV)")
    print("=" * 70)
    clf_results = train_classification(X_train, y_train, cv=5)

    print("\n" + "=" * 70)
    print("4. SELECT BEST CLASSIFIER (validation ROC-AUC)")
    print("=" * 70)
    best_name, best_model = select_best_classifier(clf_results, X_val, y_val)
    save_model(best_model, "trained_model.pkl")

    print("\n" + "=" * 70)
    print("5. EVALUATE ON HELD-OUT TEST SET")
    print("=" * 70)
    clf_metrics = evaluate_classifier(best_model, X_test, y_test, feature_cols,
                                      X_train=X_train, y_train=y_train)

    print("\n" + "=" * 70)
    print("6. FEATURE SELECTION (SelectKBest) DEMO")
    print("=" * 70)
    Xtr_k, Xva_k, Xte_k, sel, mask = select_k_best(
        X_train, y_train, X_val, X_test, k=20)
    kept = [feature_cols[i] for i in range(len(feature_cols)) if mask[i]]
    print("Top selected features:", kept[:10], "...")

    print("\n" + "=" * 70)
    print("7. SECONDARY REGRESSION DEMO (predict interest_rate)")
    print("=" * 70)
    (Xr_tr, Xr_va, Xr_te, yr_tr, yr_va, yr_te, reg_feats, _) = prepare_pipeline(
        train, test, target="interest_rate",
        exclude_cols=[], save_scaler=False,
        test_size=0.2, random_state=cfg.RANDOM_STATE, stratify=None)
    reg_results = train_regression(Xr_tr, yr_tr, cv=5)
    best_reg = None
    best_r2 = -1e9
    for name, m in reg_results.items():
        est = m.best_estimator_ if hasattr(m, "best_estimator_") else m
        pred = est.predict(Xr_te)
        r2 = regression_metrics(yr_te, pred)["R2"]
        print(f"[reg] {name}: test R2={r2:.4f}")
        if r2 > best_r2:
            best_r2, best_reg = r2, est
    reg_metrics = evaluate_regression(best_reg, Xr_te, yr_te)

    summary = {
        "best_classifier": best_name,
        "classification_metrics": clf_metrics,
        "regression_metrics": reg_metrics,
    }
    print("\n" + "=" * 70)
    print("DONE. Artifacts in models/ and reports/")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    run_full_pipeline()
