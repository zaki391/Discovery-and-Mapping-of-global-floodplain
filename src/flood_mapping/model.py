"""Model training and inference utilities for flood segmentation."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a RandomForest model for flood/non-flood pixel classification.

    Args:
        features: Feature matrix with shape (n_samples, n_features).
        labels: Binary labels with shape (n_samples,).
        random_state: Seed value for reproducibility.

    Returns:
        Trained RandomForestClassifier instance.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(features, labels)
    return model


def save_model(model: RandomForestClassifier, path: Path) -> None:
    """Persist a trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> RandomForestClassifier:
    """Load a trained RandomForest model from disk."""
    return joblib.load(path)
