"""Unit tests for evaluation metrics."""

import numpy as np

from flood_mapping.evaluation import classification_metrics, iou_score


def test_iou_score_perfect_overlap() -> None:
    """IoU should be 1.0 when masks are identical."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    assert iou_score(y_true, y_pred) == 1.0


def test_classification_metrics_keys_exist() -> None:
    """Metrics output should include all expected portfolio metrics."""
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])

    metrics = classification_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {"iou", "accuracy", "precision", "recall"}
