"""Data ingestion utilities for loading geospatial raster and vector data."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio


def load_raster(path: Path) -> tuple[np.ndarray, dict]:
    """Load a multi-band raster as a NumPy array and return metadata.

    Args:
        path: File path to a GeoTIFF or raster file.

    Returns:
        Tuple containing an array with shape (bands, height, width)
        and the raster metadata dictionary.
    """
    with rasterio.open(path) as src:
        image = src.read()
        metadata = src.meta.copy()
    return image, metadata


def load_vector_labels(path: Path) -> gpd.GeoDataFrame:
    """Load vector flood labels from a geospatial file.

    Args:
        path: Path to a vector dataset (GeoJSON, SHP, GPKG).

    Returns:
        GeoDataFrame containing geometry and target attributes.
    """
    return gpd.read_file(path)
