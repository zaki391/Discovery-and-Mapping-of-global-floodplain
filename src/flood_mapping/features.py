"""Feature engineering methods for flood extent modeling."""

from __future__ import annotations

import numpy as np


def compute_ndwi(green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Water Index (NDWI).

    Args:
        green_band: Green channel array.
        nir_band: Near-infrared channel array.

    Returns:
        NDWI raster where higher values indicate potential water.
    """
    denominator = green_band + nir_band
    denominator = np.where(denominator == 0, 1e-6, denominator)
    return (green_band - nir_band) / denominator


def stack_features(image: np.ndarray) -> np.ndarray:
    """Create a pixel-wise feature matrix from raster bands.

    Args:
        image: Array with shape (bands, height, width).

    Returns:
        Flattened feature matrix with shape (pixels, features).
    """
    bands, height, width = image.shape
    return image.reshape(bands, height * width).T
