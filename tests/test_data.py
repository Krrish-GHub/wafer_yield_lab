from pathlib import Path

from wafer_yield_lab.data import load_dataset


def test_load_dataset_maps_fail_timestamp_and_missing(tmp_path: Path):
    features = tmp_path / "secom.data"
    labels = tmp_path / "labels.data"
    features.write_text(" ".join(["1"] * 589 + ["NaN"]) + "\n" + " ".join(["4"] * 590) + "\n", encoding="utf-8")
    labels.write_text("-1 2008-01-01\n1 2008-01-02\n", encoding="utf-8")
    X, y, timestamps = load_dataset(features, labels)
    assert X.shape == (2, 590)
    assert X.iloc[0, -1] != X.iloc[0, -1]
    assert y.tolist() == [0, 1]
    assert timestamps.tolist() == ["2008-01-01", "2008-01-02"]
