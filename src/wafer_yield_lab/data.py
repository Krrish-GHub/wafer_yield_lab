"""Loading and validating the public UCI SECOM text files."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
RAW_DIR = Path("data/raw")


def download_dataset(destination: Path = RAW_DIR) -> tuple[Path, Path]:
    """Download and safely extract the two UCI data files; return their paths."""
    destination.mkdir(parents=True, exist_ok=True)
    response = requests.get(UCI_ZIP_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
        feature_name = next((name for name in names if name.endswith("secom.data")), None)
        label_name = next((name for name in names if "secom_labels" in name.lower()), None)
        if not feature_name or not label_name:
            raise ValueError(f"UCI archive did not contain expected SECOM files: {names}")
        # Extract only the expected data files, avoiding arbitrary archive paths.
        features_path = destination / "secom.data"
        labels_path = destination / "secom_labels.data"
        features_path.write_bytes(archive.read(feature_name))
        labels_path.write_bytes(archive.read(label_name))
    return features_path, labels_path


def load_dataset(features_path: Path, labels_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load 590 anonymized signals, binary pass/fail target, and source timestamp."""
    features = pd.read_csv(features_path, sep=r"\s+", header=None, na_values="NaN")
    labels = pd.read_csv(labels_path, sep=r"\s+", header=None, names=["target", "timestamp"])
    if len(features) != len(labels) or features.shape[1] != 590:
        raise ValueError(f"Unexpected SECOM shape: features={features.shape}, labels={labels.shape}")
    if not set(labels["target"].unique()).issubset({-1, 1}):
        raise ValueError("SECOM target must contain only -1 (pass) and 1 (fail)")
    # The source records -1 = pass, +1 = fail. Use 1 for the operational positive class: fail.
    y = (labels["target"] == 1).astype("int8").rename("fail")
    X = features.astype("float64")
    X.columns = [f"sensor_{index:03d}" for index in range(X.shape[1])]
    return X, y, labels["timestamp"].astype(str).rename("timestamp")
