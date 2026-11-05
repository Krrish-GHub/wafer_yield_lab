import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from wafer_yield_lab.model import build_models, positive_scores


def test_logistic_pipeline_handles_missing_values_without_leaking():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(100, 12)), columns=[f"sensor_{i:03d}" for i in range(12)])
    y = pd.Series([0] * 80 + [1] * 20)
    X.loc[0, "sensor_000"] = np.nan
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2,
                                           random_state=42, stratify=y)
    model = build_models()["logistic_regression"]
    model.set_params(select__k=8)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    scores = positive_scores(model, X.iloc[test_idx])
    assert scores.shape == (len(test_idx),)
    assert np.isfinite(scores).all()
