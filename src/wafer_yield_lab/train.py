"""Train and compare models, then write reproducible report and static demo assets."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .data import RAW_DIR, download_dataset, load_dataset
from .model import (
    build_models,
    export_logistic,
    f1_threshold,
    positive_scores,
    score_predictions,
    threshold_table,
)

SEED = 42


def train(output_dir: Path = Path("reports"), site_dir: Path = Path("site"),
          features_path: Path | None = None, labels_path: Path | None = None) -> dict:
    if features_path is None or labels_path is None:
        features_path, labels_path = download_dataset(RAW_DIR)
    X, y, timestamps = load_dataset(features_path, labels_path)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED,
                                           stratify=y)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    report = {
        "threshold_selection": {
            "method": "maximum fail-class F1 on 5-fold out-of-fold training scores",
            "seed": SEED,
            "tie_break": "highest threshold",
            "warning": "Threshold is a study convention, not a factory cost policy."
        },
        "dataset": {
            "source": "UCI SECOM",
            "source_url": "https://archive.ics.uci.edu/dataset/179/secom",
            "citation": "McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305.",
            "rows": len(X), "sensors": int(X.shape[1]),
            "pass_count": int((y == 0).sum()), "fail_count": int((y == 1).sum()),
            "positive_definition": "fail (source label +1)",
            "missing_cells": int(X.isna().sum().sum()),
            "train_rows": len(train_idx), "test_rows": len(test_idx),
            "train_fails": int(y_train.sum()), "test_fails": int(y_test.sum()),
        },
        "split": {"method": "stratified random holdout", "test_fraction": 0.2,
                  "seed": SEED, "warning": "One small holdout is noisy; not a production or future-time guarantee."},
        "models": {},
        "thresholds": {},
    }
    fitted = {}
    scores = {}
    thresholds = {"majority_baseline": 0.5}
    for name, estimator in build_models(SEED).items():
        if name == "majority_baseline":
            threshold = 0.5
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            oof = cross_val_predict(estimator, X_train, y_train, cv=cv,
                                    method="predict_proba", n_jobs=1)[:, 1]
            threshold = f1_threshold(y_train.to_numpy(), oof)
        estimator.fit(X_train, y_train)
        fitted[name] = estimator
        thresholds[name] = threshold
        score = positive_scores(estimator, X_test)
        scores[name] = score
        report["models"][name] = score_predictions(y_test.to_numpy(), score, threshold)
    report["threshold_selection"]["model_thresholds"] = thresholds
    for name in ("logistic_regression", "extra_trees", "hist_gradient_boosting"):
        report["thresholds"][name] = threshold_table(y_test.to_numpy(), scores[name])
    # The demo is intentionally a static inspection of held-out examples, not a live endpoint.
    logistic = fitted["logistic_regression"]
    browser_model = export_logistic(logistic, str(site_dir / "model.json"))
    browser_model["threshold"] = thresholds["logistic_regression"]
    (site_dir / "model.json").write_text(json.dumps(browser_model, indent=2) + "\n", encoding="utf-8")
    sample_rows = []
    for idx in test_idx[:24]:
        sample_rows.append({
            "id": f"holdout-{int(idx):04d}",
            "actual": "fail" if int(y.iloc[idx]) else "pass",
            "timestamp": str(timestamps.iloc[idx]),
            "features": [None if pd.isna(v) else float(v) for v in X.iloc[idx].tolist()],
            "score": float(positive_scores(logistic, X.iloc[[idx]])[0]),
        })
    (site_dir / "samples.json").write_text(json.dumps(sample_rows, separators=(",", ":")) + "\n", encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    joblib.dump({"model": logistic, "feature_names": list(X.columns), "seed": SEED}, output_dir / "logistic_model.joblib")
    return report


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
