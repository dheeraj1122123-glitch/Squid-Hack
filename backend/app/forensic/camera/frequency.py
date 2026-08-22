"""Frequency domain analysis."""
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.fft import fft2, fftshift
from scipy.stats import entropy

from app.forensic.preprocessing.normalization import to_float01


def radial_profile(spectrum: np.ndarray, center: tuple[int, int] | None = None) -> np.ndarray:
    h, w = spectrum.shape
    cy, cx = center or (h // 2, w // 2)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cy, cx, h - cy, w - cx)
    profile = np.zeros(max_r)
    for i in range(max_r):
        mask = r == i
        if mask.any():
            profile[i] = spectrum[mask].mean()
    return profile


def frequency_analysis(img: np.ndarray, artifact_dir: Path | None = None) -> dict[str, Any]:
    f = to_float01(img)
    if f.ndim == 3:
        gray = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
    else:
        gray = f

    fft = fftshift(fft2(gray))
    magnitude = np.log1p(np.abs(fft))
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    low = magnitude[cy - h // 8 : cy + h // 8, cx - w // 8 : cx + w // 8]
    mid_mask = np.ones_like(magnitude, dtype=bool)
    mid_mask[cy - h // 8 : cy + h // 8, cx - w // 8 : cx + w // 8] = False
    mid_mask[cy - h // 4 : cy + h // 4, cx - w // 4 : cx + w // 4] = True
    mid = magnitude[mid_mask]
    high_mask = ~mid_mask
    high_mask[cy - h // 4 : cy + h // 4, cx - w // 4 : cx + w // 4] = False
    high = magnitude[high_mask]

    profile = radial_profile(magnitude)
    hist, _ = np.histogram(magnitude.ravel(), bins=256, density=True)
    hist = hist[hist > 0]
    spec_entropy = float(entropy(hist))

    artifact_path = None
    if artifact_dir:
        artifact_path = artifact_dir / "frequency_spectrum.png"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        norm = magnitude - magnitude.min()
        if norm.max() > 0:
            norm = norm / norm.max()
        cv2.imwrite(str(artifact_path), (norm * 255).astype(np.uint8))

    return {
        "status": "success",
        "availability": True,
        "low_frequency_energy": float(np.mean(low)),
        "mid_frequency_energy": float(np.mean(mid)) if mid.size else 0.0,
        "high_frequency_energy": float(np.mean(high)) if high.size else 0.0,
        "spectral_entropy": spec_entropy,
        "radial_profile_mean": float(np.mean(profile)),
        "radial_profile_std": float(np.std(profile)),
        "artifact_path": str(artifact_path) if artifact_path else None,
    }
