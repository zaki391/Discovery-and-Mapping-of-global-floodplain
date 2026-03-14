"""Configuration objects for flood mapping experiments."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    """Holds run-time configuration for the flood mapping pipeline."""

    input_raster_path: Path
    label_raster_path: Path
    model_output_path: Path = Path("models/flood_segmenter.joblib")
    random_state: int = 42
    test_size: float = 0.2
    threshold: float = 0.5
