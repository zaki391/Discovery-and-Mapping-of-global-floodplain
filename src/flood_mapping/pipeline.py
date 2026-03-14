"""End-to-end flood mapping pipeline orchestrator."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from sklearn.model_selection import train_test_split

from flood_mapping.config import PipelineConfig
from flood_mapping.data_ingestion import load_raster
from flood_mapping.evaluation import classification_metrics
from flood_mapping.features import stack_features
from flood_mapping.inference import predict_mask
from flood_mapping.model import save_model, train_random_forest
from flood_mapping.preprocessing import normalize_bands


class FloodMappingPipeline:
    """Runs ingestion, preprocessing, training, prediction, and evaluation."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self) -> dict[str, float]:
        """Execute the full training/evaluation pipeline and return metrics."""
        image, _ = load_raster(self.config.input_raster_path)
        labels, _ = load_raster(self.config.label_raster_path)

        image = normalize_bands(image)
        feature_matrix = stack_features(image)
        label_vector = labels[0].reshape(-1).astype(np.uint8)

        x_train, x_val, y_train, y_val = train_test_split(
            feature_matrix,
            label_vector,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=label_vector,
        )

        model = train_random_forest(x_train, y_train, random_state=self.config.random_state)
        save_model(model, self.config.model_output_path)

        y_pred = model.predict(x_val)
        metrics = classification_metrics(y_val, y_pred)
        return metrics

    def predict_full_scene(self) -> np.ndarray:
        """Train quickly and return full-scene predictions for visualization demos."""
        image, _ = load_raster(self.config.input_raster_path)
        labels, _ = load_raster(self.config.label_raster_path)

        image = normalize_bands(image)
        features = stack_features(image)
        label_vector = labels[0].reshape(-1).astype(np.uint8)

        model = train_random_forest(features, label_vector, random_state=self.config.random_state)
        height, width = labels.shape[1], labels.shape[2]
        return predict_mask(model, features, (height, width))

    def describe(self) -> dict:
        """Expose configuration values for logging/debugging purposes."""
        return asdict(self.config)
