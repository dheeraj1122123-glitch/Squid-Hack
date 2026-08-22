"""Patch extraction for forensic analysis."""
from typing import Iterator

import numpy as np


def extract_patches(
    img: np.ndarray,
    patch_size: int = 128,
    stride: int | None = None,
) -> Iterator[tuple[np.ndarray, tuple[int, int]]]:
    if stride is None:
        stride = patch_size // 2
    h, w = img.shape[:2]
    for y in range(0, max(1, h - patch_size + 1), stride):
        for x in range(0, max(1, w - patch_size + 1), stride):
            patch = img[y : y + patch_size, x : x + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                yield patch, (y, x)


def patch_grid(img: np.ndarray, patch_size: int = 64, stride: int | None = None) -> list[tuple[np.ndarray, int, int]]:
    return [(p, y, x) for p, (y, x) in extract_patches(img, patch_size, stride)]
