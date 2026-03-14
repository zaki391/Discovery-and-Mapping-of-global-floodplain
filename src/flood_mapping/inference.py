"""Prediction helpers for converting model outputs into flood maps."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def predict_mask(
    model: RandomForestClassifier,
    features: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Generate flood mask predictions and reshape to raster layout.

    Args:
        model: Trained classifier.
        features: Pixel feature matrix with shape (pixels, features).
        image_shape: Spatial raster shape (height, width).

    Returns:
        Predicted binary flood mask with shape (height, width).
    """
    prediction = model.predict(features)
    return prediction.reshape(image_shape)
