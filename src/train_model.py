"""Model training: multiple supervised algorithms + hyperparameter tuning.

Classification (primary task - loan approval):
    Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Naive Bayes

Regression (secondary demo - predict interest_rate):
    Linear Regression, Decision Tree, Random Forest
"""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Dict, Tuple

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import config as cfg

MODELS_DIR = cfg.MODELS_DIR


def get_classification_models() -> Dict[str, Tuple]:
    """Return dict of model_name -> (estimator, param_grid)."""
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=cfg.RANDOM_STATE, class_weight="balanced"),
            {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=cfg.RANDOM_STATE, class_weight="balanced"),
            {"n_estimators": [100, 200], "max_depth": [None, 10, 20],
             "min_samples_split": [2, 5]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [5, 9, 15], "weights": ["uniform", "distance"]},
        ),
        "SVM": (
            CalibratedClassifierCV(
                SVC(class_weight="balanced"), ensemble=False, cv=3),
            {"estimator__C": [1.0, 10.0], "estimator__gamma": ["scale"]},
        ),
        "NaiveBayes": (
            GaussianNB(),
            {"var_smoothing": [1e-9, 1e-8, 1e-7]},
        ),
    }


def get_regression_models() -> Dict[str, Tuple]:
    return {
        "LinearRegression": (LinearRegression(), {}),
        "DecisionTreeRegressor": (
            DecisionTreeRegressor(random_state=cfg.RANDOM_STATE),
            {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
        ),
        "RandomForestRegressor": (
            RandomForestRegressor(random_state=cfg.RANDOM_STATE),
            {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
        ),
    }


def train_classification(X_train, y_train, cv: int = 5, scoring: str = "roc_auc",
                         n_jobs: int = -1) -> Dict[str, GridSearchCV]:
    results = {}
    for name, (est, grid) in get_classification_models().items():
        if grid:
            gs = GridSearchCV(est, grid, cv=cv, scoring=scoring,
                              n_jobs=n_jobs, refit=True)
            gs.fit(X_train, y_train)
            results[name] = gs
            print(f"[train] {name}: best CV {scoring}={gs.best_score_:.4f} "
                  f"| params={gs.best_params_}")
        else:
            est.fit(X_train, y_train)
            cvs = cross_val_score(est, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs)
            results[name] = est
            print(f"[train] {name}: mean CV {scoring}={cvs.mean():.4f} (+/- {cvs.std():.4f})")
    return results


def train_regression(X_train, y_train, cv: int = 5, n_jobs: int = -1) -> Dict[str, GridSearchCV]:
    results = {}
    for name, (est, grid) in get_regression_models().items():
        if grid:
            gs = GridSearchCV(est, grid, cv=cv, scoring="r2", n_jobs=n_jobs, refit=True)
            gs.fit(X_train, y_train)
            results[name] = gs
            print(f"[train] {name}: best CV r2={gs.best_score_:.4f}")
        else:
            est.fit(X_train, y_train)
            cvs = cross_val_score(est, X_train, y_train, cv=cv, scoring="r2", n_jobs=n_jobs)
            results[name] = est
            print(f"[train] {name}: mean CV r2={cvs.mean():.4f}")
    return results


def select_best_classifier(results: Dict, X_val, y_val) -> Tuple[str, object]:
    from sklearn.metrics import roc_auc_score
    best_name, best_score, best_model = None, -1, None
    for name, model in results.items():
        est = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        proba = est.predict_proba(X_val)[:, 1] if hasattr(est, "predict_proba") else None
        if proba is not None:
            score = roc_auc_score(y_val, proba)
        else:
            score = est.score(X_val, y_val)
        print(f"[select] {name}: val roc_auc={score:.4f}")
        if score > best_score:
            best_score, best_name, best_model = score, name, est
    print(f"[select] BEST classifier = {best_name} (val roc_auc={best_score:.4f})")
    return best_name, best_model


def save_model(model, name: str = "trained_model.pkl", model_dir: Path = MODELS_DIR) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / name
    joblib.dump(model, path)
    print(f"[save] model -> {path}")
    return path


def load_model(name: str = "trained_model.pkl", model_dir: Path = MODELS_DIR):
    path = model_dir / name
    return joblib.load(path)
