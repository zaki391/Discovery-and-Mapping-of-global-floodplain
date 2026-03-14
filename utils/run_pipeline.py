"""CLI entry point for running the flood mapping pipeline."""

from __future__ import annotations

from pathlib import Path

from flood_mapping.config import PipelineConfig
from flood_mapping.pipeline import FloodMappingPipeline


def main() -> None:
    """Run training/evaluation using example data paths."""
    config = PipelineConfig(
        input_raster_path=Path("data/raw/example_satellite.tif"),
        label_raster_path=Path("data/raw/example_flood_labels.tif"),
    )
    pipeline = FloodMappingPipeline(config)
    metrics = pipeline.run()
    print("Pipeline metrics:", metrics)


if __name__ == "__main__":
    main()
