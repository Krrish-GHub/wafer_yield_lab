import numpy as np

from wafer_yield_lab.model import score_predictions, threshold_table


def test_fail_is_positive_and_confusion_counts_are_explicit():
    result = score_predictions(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.7, 0.2]), 0.5)
    assert result["confusion_matrix"] == {"true_pass": 1, "false_alarm": 1,
                                          "missed_fail": 1, "caught_fail": 1}
    assert result["recall_fail"] == 0.5


def test_threshold_table_is_monotonic_for_recall():
    rows = threshold_table(np.array([0, 0, 1, 1]), np.array([0.1, 0.3, 0.6, 0.9]))
    recalls = [row["recall_fail"] for row in rows]
    assert recalls == sorted(recalls, reverse=True)
