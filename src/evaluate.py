"""Model evaluation and performance visualization.

Classification metrics: accuracy, precision, recall, f1, roc-auc
Regression metrics:   MAE, MSE, RMSE, R2

Visualizations saved to reports/:
    confusion_matrix.png, roc_curve.png, precision_recall_curve.png,
    feature_importance.png, learning_curve.png,
    prediction_vs_actual.png, residual_plot.png

Note: this module does NOT force a matplotlib backend so it works both headless
(pipeline) and interactively (notebook, where plots render inline). Headless
entry points (e.g. pipeline.py) set `matplotlib.use("Agg")` themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import learning_curve

import config as cfg

REPORTS_DIR = cfg.REPORTS_DIR
sns.set_theme(style="whitegrid")
COLORS = sns.color_palette("viridis", 8)


def classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """Compute standard classification metrics (optionally including ROC-AUC)."""
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        m["roc_auc"] = roc_auc_score(y_true, y_proba)
    return m


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute standard regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def save_classification_report(y_true, y_pred, y_proba, labels=(0, 1),
                               path: Path = REPORTS_DIR / "classification_report.txt") -> None:
    """Write a human-readable classification report (metrics + confusion matrix)."""
    report = classification_report(y_true, y_pred, labels=labels,
                                    target_names=["Rejected", "Approved"])
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = classification_metrics(y_true, y_pred, y_proba)
    header = "=" * 60 + "\n"
    header += "LOAN APPROVAL - CLASSIFICATION REPORT\n"
    header += "=" * 60 + "\n\n"
    header += "Key Metrics\n" + "-" * 60 + "\n"
    header += "\n".join(f"{k:>10}: {v:.4f}" for k, v in metrics.items()) + "\n\n"
    header += "Confusion Matrix (rows=true, cols=pred)\n" + "-" * 60 + "\n"
    header += f"           Predicted=0  Predicted=1\n"
    header += f"Actual=0   {cm[0,0]:>10}  {cm[0,1]:>10}\n"
    header += f"Actual=1   {cm[1,0]:>10}  {cm[1,1]:>10}\n\n"
    content = header + "Detailed Report\n" + "-" * 60 + "\n" + report
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"[report] saved -> {path}")


def plot_confusion_matrix(y_true, y_pred, path: Path = REPORTS_DIR / "confusion_matrix.png") -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[plot] saved -> {path}")


def plot_roc_curve(y_true, y_proba, path: Path = REPORTS_DIR / "roc_curve.png") -> float:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color=COLORS[0], lw=2, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.title("ROC Curve"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[plot] saved -> {path}")
    return auc


def plot_precision_recall(y_true, y_proba, path: Path = REPORTS_DIR / "precision_recall_curve.png") -> None:
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, color=COLORS[2], lw=2, label=f"PR (AP={ap:.3f})")
    plt.title("Precision-Recall Curve"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(loc="lower left"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[plot] saved -> {path}")


def plot_feature_importance(model, feature_names, top: int = 20,
                            path: Path = REPORTS_DIR / "feature_importance.png") -> None:
    """Plot the top feature importances/coefficients if the model exposes them."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    else:
        print("[plot] model exposes no importances/coef_ -> skipping feature importance")
        return
    order = np.argsort(importances)[::-1][:top]
    names = [feature_names[i] for i in order]
    vals = importances[order]
    plt.figure(figsize=(8, max(4, top * 0.35)))
    sns.barplot(x=vals, y=names, hue=names, palette="viridis", legend=False)
    plt.title("Feature Importance (Top %d)" % len(names))
    plt.xlabel("Importance"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[plot] saved -> {path}")


def plot_learning_curve(estimator, X, y, cv: int = 5,
                        path: Path = REPORTS_DIR / "learning_curve.png") -> None:
    """Plot a learning curve. Pass TRAINING data so it reflects fitting dynamics."""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)
    tr_mean, tr_std = train_scores.mean(1), train_scores.std(1)
    va_mean, va_std = val_scores.mean(1), val_scores.std(1)
    plt.figure(figsize=(6, 4))
    plt.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15)
    plt.fill_between(train_sizes, va_mean - va_std, va_mean + va_std, alpha=0.15)
    plt.plot(train_sizes, tr_mean, "o-", label="Training")
    plt.plot(train_sizes, va_mean, "o-", label="Validation")
    plt.title("Learning Curve (ROC-AUC)"); plt.xlabel("Training examples"); plt.ylabel("Score")
    plt.legend(loc="best"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[plot] saved -> {path}")


def plot_prediction_vs_actual(y_true, y_pred,
                              path: Path = REPORTS_DIR / "prediction_vs_actual.png") -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.4, color=COLORS[1])
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lim, lim, "--", color="gray")
    plt.title("Prediction vs Actual"); plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.xlim(lim); plt.ylim(lim); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[plot] saved -> {path}")


def plot_residuals(y_true, y_pred, path: Path = REPORTS_DIR / "residual_plot.png") -> None:
    resid = np.asarray(y_true) - np.asarray(y_pred)
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, resid, alpha=0.4, color=COLORS[3])
    plt.axhline(0, color="gray", ls="--")
    plt.title("Residual Plot"); plt.xlabel("Predicted"); plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[plot] saved -> {path}")


def evaluate_classifier(model, X_test, y_test, feature_names,
                        X_train=None, y_train=None,
                        threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate the best classifier on the held-out test set and produce all artifacts."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = classification_metrics(y_test, pred, proba)

    plot_confusion_matrix(y_test, pred)
    plot_roc_curve(y_test, proba)
    plot_precision_recall(y_test, proba)
    plot_feature_importance(model, feature_names)
    try:
        # Learning curve should reflect the training set, not the test set.
        lc_X, lc_y = (X_train, y_train) if X_train is not None else (X_test, y_test)
        plot_learning_curve(model, lc_X, lc_y)
    except Exception as exc:  # learning curve is best-effort
        print(f"[plot] learning curve skipped: {exc}")
    save_classification_report(y_test, pred, proba)

    for k, v in metrics.items():
        print(f"[metric] {k}: {v:.4f}")
    return metrics


def evaluate_regression(model, X_test, y_test) -> Dict[str, float]:
    """Evaluate a regressor on the held-out test set and produce plots."""
    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)
    plot_prediction_vs_actual(y_test, pred)
    plot_residuals(y_test, pred)
    for k, v in metrics.items():
        print(f"[metric] {k}: {v:.4f}")
    return metrics
