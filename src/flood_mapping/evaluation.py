"""Evaluation metrics for flood segmentation quality checks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def iou_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Intersection-over-Union for binary masks."""
    intersection = np.logical_and(y_true == 1, y_pred == 1).sum()
    union = np.logical_or(y_true == 1, y_pred == 1).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return common binary classification metrics for flood predictions."""
    return {
        "iou": iou_score(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
