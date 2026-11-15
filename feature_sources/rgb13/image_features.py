"""Spatial RGB features for a single pseudo-color thermal image."""

from __future__ import annotations

import numpy as np
from PIL import Image


RGB_GRID_COLUMNS = [
    f"img_grid_r{row}_c{column}_{channel}"
    for row in range(1, 3)
    for column in range(1, 3)
    for channel in ("red", "green", "blue")
]


def extract_rgb_grid(image: Image.Image) -> list[float]:
    """Return mean RGB values in [0, 1] for each of nine image regions."""
    pixels = np.asarray(image.convert("RGB"))
    height, width = pixels.shape[:2]
    if height < 3 or width < 3:
        # Ensure every region contains pixels even for a very small image.
        pixels = np.pad(
            pixels,
            ((0, max(0, 2 - height)), (0, max(0, 2 - width)), (0, 0)),
            mode="edge",
        )

    features = []
    for row in np.array_split(pixels, 2, axis=0):
        for region in np.array_split(row, 2, axis=1):
            features.extend((region.mean(axis=(0, 1)) / 255.0).tolist())
    return features
