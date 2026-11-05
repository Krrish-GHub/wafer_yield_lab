"""Leakage-safe model pipelines and evaluation helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(seed: int = 42) -> dict[str, Any]:
    """Return fixed, reproducible baselines; every learned transform fits on train only."""
    preprocess = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("nonconstant", VarianceThreshold(threshold=0.0)),
    ]
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(preprocess + [
            ("select", SelectKBest(score_func=f_classif, k=40)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", C=0.1,
                                          max_iter=5000, random_state=seed)),
        ]),
        "extra_trees": Pipeline(preprocess + [
            ("model", ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2,
                                           class_weight="balanced", max_features="sqrt",
                                           random_state=seed, n_jobs=-1)),
        ]),
        "hist_gradient_boosting": Pipeline(preprocess + [
            ("model", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                                                      max_leaf_nodes=15, l2_regularization=1.0,
                                                      random_state=seed)),
        ]),
    }


def score_predictions(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Metrics that keep the rare fail class visible; scores are fail probabilities/scores."""
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "average_precision": float(average_precision_score(y_true, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_fail": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_fail": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_fail": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"true_pass": int(tn), "false_alarm": int(fp),
                             "missed_fail": int(fn), "caught_fail": int(tp)},
    }


def positive_scores(estimator: Any, X: Any) -> np.ndarray:
    """Get positive-class probability, or a monotonic decision score for baselines."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def threshold_table(y_true: np.ndarray, scores: np.ndarray,
                    thresholds: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.35, 0.50)) -> list[dict[str, Any]]:
    return [score_predictions(y_true, scores, threshold=t) for t in thresholds]


def export_logistic(estimator: Pipeline, output_path: str) -> dict[str, Any]:
    """Export the fitted logistic pipeline as a compact, dependency-free browser model."""
    import json
    from pathlib import Path

    imputer = estimator.named_steps["impute"]
    variance = estimator.named_steps["nonconstant"]
    scaler = estimator.named_steps["scale"]
    selector = estimator.named_steps["select"]
    classifier = estimator.named_steps["model"]
    variance_indices = np.flatnonzero(variance.get_support())
    selected_indices = variance_indices[selector.get_support()]
    model = {
        "model": "standardized logistic regression",
        "positive_label": "fail",
        "sensor_count": 590,
        "selected_indices": [int(i) for i in selected_indices],
        "medians": [float(imputer.statistics_[i]) if np.isfinite(imputer.statistics_[i]) else 0.0
                    for i in selected_indices],
        "means": [float(x) for x in scaler.mean_],
        "scales": [float(x) for x in scaler.scale_],
        "coefficients": [float(x) for x in classifier.coef_[0]],
        "intercept": float(classifier.intercept_[0]),
        "threshold": 0.5,
        "feature_names": [f"sensor_{i:03d}" for i in selected_indices],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    return model


def f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose an F1-maximizing threshold using training-only out-of-fold scores."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(f1 == f1.max())
    # On ties prefer the highest threshold, reducing the flagged volume.
    return float(thresholds[best[-1]])
