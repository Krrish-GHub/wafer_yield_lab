# Wafer Yield Lab

A reproducible machine-learning study of semiconductor process quality: can anonymized sensor signals help flag a unit that later fails its in-house yield check?

[Open the interactive demo](https://krrish-ghub.github.io/wafer-yield-lab/) · [View evaluation report](reports/metrics.json)

## Why this is a useful ML problem

In a real quality-control workflow, a model that catches more failing units may also raise more false alarms. This project makes that trade-off visible instead of hiding the rare fail class behind overall accuracy. It compares a no-skill baseline, a regularized and interpretable logistic model, Extra Trees and histogram gradient boosting. The dataset is small, high-dimensional, imbalanced and incomplete—exactly the kind of conditions that make validation and preprocessing choices matter.

This is an educational offline experiment, not a factory control system or a model suitable for production decisions.

## Dataset and responsible use

The project uses the [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom), credited to McCann and Johnston (2008). UCI describes 1,567 production entities, 590 anonymized process measurements, pass/fail labels and timestamps; its source label `-1` means pass and `+1` means fail. The dataset contains missing values and only 104 failures. It is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

No sensor has a public engineering meaning in the dataset, so names such as `sensor_017` are identifiers only. The public dataset is historical; this one random holdout cannot establish performance on future production, other lines or changed process conditions. A real deployment would need time-based and external validation, process-engineer review, drift monitoring, and an explicit cost model for missed failures versus false alarms. Do not use the demo to accept or reject real products.

## Results

The fixed-seed stratified split holds out 314 records, including 21 failures. Every preprocessing step is inside the estimator pipeline and fit on training folds only. Each model's reported operating threshold is selected by maximizing fail-class F1 over five-fold out-of-fold scores on the training split; the test set is used only for evaluation. Threshold selection is a transparent study convention, not a production cost policy.

| Model | Test ROC-AUC | Test average precision | CV-selected threshold | Fail recall | Fail precision | False alarms |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.5000 | 0.0669 | 0.500 | 0.000 | 0.000 | 0 |
| Logistic regression | 0.7468 | 0.1748 | 0.663 | 0.429 | 0.214 | 33 |
| Extra Trees | 0.7743 | 0.2205 | 0.100 | 0.571 | 0.136 | 76 |
| Histogram gradient boosting | 0.7809 | 0.2001 | 0.018 | 0.333 | 0.212 | 26 |

These results show the trade-off rather than a production-ready winner: Extra Trees caught 12 of 21 held-out failures but also flagged 76 passing examples; logistic regression caught 9 and flagged 33. The test holdout is small, and model choice and threshold performance can vary substantially. The complete report includes confusion counts and a threshold sweep. It is an exploratory comparison, not a hyperparameter search or a reliable ranking.

The static demo uses the fitted logistic model on 24 selected held-out public examples and shows the largest signed feature contributions for each one. The browser implementation matches the Python scores to floating-point precision. Inputs stay in the browser; no submitted data is sent to a server. Sensor IDs are anonymized, so those contributions have no direct process-engineering interpretation.

## Run it locally

Requires Python 3.10+ and internet access for the one-time UCI dataset download.

```bash
uv sync --extra dev
uv run python -m wafer_yield_lab.train
uv run pytest -q
uv run ruff check .
uv run python -m http.server 8000 --directory site
```

Open <http://localhost:8000> for the local static demo. Training downloads UCI's archive into `data/raw/` (excluded from Git) and writes the reproducible report to `reports/metrics.json`, the fitted logistic model to `reports/logistic_model.joblib`, and the browser model/sample bundle to `site/`.

The checked-in browser bundle is a compact logistic model plus 24 selected, labeled holdout examples, so the demo works immediately without downloading the full dataset. To refresh it after retraining, run the training command; then commit the updated `site/model.json`, `site/samples.json` and metrics report if the experiment intentionally changes.

## Repository map

- `src/wafer_yield_lab/data.py` downloads and validates the original UCI files.
- `src/wafer_yield_lab/model.py` defines leakage-safe pipelines, metrics and the small browser-model export.
- `src/wafer_yield_lab/train.py` makes the seeded split, fits models and records the report.
- `tests/` covers label orientation, confusion counts, threshold behavior and missing-value handling.
- `site/` is the static, GitHub Pages-compatible demo.

## Citation

McCann, M. & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C54305>.

## License

Project source code: MIT. Dataset: CC BY 4.0; please retain the UCI attribution above.
