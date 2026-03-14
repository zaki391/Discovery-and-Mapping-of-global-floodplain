"""Visualization tools for flood mapping outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_flood_detection_map(mask: np.ndarray, output_path: Path) -> None:
    """Create a flood detection map visualization from binary mask."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap="Blues")
    plt.title("Flood Detection Map")
    plt.colorbar(label="Flood probability / class")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_before_after(before: np.ndarray, after: np.ndarray, output_path: Path) -> None:
    """Save side-by-side before/after satellite comparison images."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(before)
    axes[0].set_title("Before Flood")
    axes[0].axis("off")

    axes[1].imshow(after)
    axes[1].set_title("After Flood")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_prediction_overlay(
    rgb_image: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    alpha: float = 0.35,
) -> None:
    """Overlay predicted flood mask on top of an RGB satellite image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(rgb_image)
    plt.imshow(mask, cmap="Reds", alpha=alpha)
    plt.title("Flood Prediction Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
