"""Preprocessing methods for geospatial imagery used in flood mapping."""

from __future__ import annotations

import cv2
import numpy as np


def normalize_bands(image: np.ndarray) -> np.ndarray:
    """Normalize each raster band to [0, 1].

    Args:
        image: Array with shape (bands, height, width).

    Returns:
        Normalized array with the same shape.
    """
    normalized = np.zeros_like(image, dtype=np.float32)
    for band_idx, band in enumerate(image):
        band_min = float(band.min())
        band_max = float(band.max())
        denominator = (band_max - band_min) or 1.0
        normalized[band_idx] = (band - band_min) / denominator
    return normalized


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a multi-band raster to a target spatial size.

    Args:
        image: Array with shape (bands, height, width).
        size: Tuple (width, height).

    Returns:
        Resized array with shape (bands, height, width).
    """
    resized_bands = [cv2.resize(band, size, interpolation=cv2.INTER_AREA) for band in image]
    return np.asarray(resized_bands, dtype=np.float32)
