"""PRNU-style fingerprint estimation and correlation."""
from pathlib import Path
from typing import Any

import numpy as np

from app.forensic.camera.noise import extract_noise_residual
from app.forensic.preprocessing.normalization import to_float01


def normalized_residual(img: np.ndarray) -> np.ndarray:
    """Compute normalized noise residual W = (I - F(I)) / sigma."""
    f = to_float01(img)
    residual = extract_noise_residual(img, method="gaussian")
    sigma = np.std(residual)
    if sigma < 1e-10:
        sigma = 1e-10
    if f.ndim == 2:
        return residual / sigma
    out = np.zeros_like(residual)
    for c in range(residual.shape[2]):
        s = np.std(residual[:, :, c])
        if s < 1e-10:
            s = 1e-10
        out[:, :, c] = residual[:, :, c] / s
    return out


def aggregate_fingerprint(images: list[np.ndarray]) -> np.ndarray:
    """Aggregate normalized residuals from reference images."""
    if not images:
        raise ValueError("No reference images provided")
    accum = None
    for img in images:
        w = normalized_residual(img)
        if w.shape[0] < 64 or w.shape[1] < 64:
            continue
        if accum is None:
            accum = w.astype(np.float64)
        else:
            h = min(accum.shape[0], w.shape[0])
            ww = min(accum.shape[1], w.shape[1])
            accum = accum[:h, :ww] + w[:h, :ww]
    if accum is None:
        raise ValueError("No valid reference images")
    return accum / len(images)


def prnu_correlation(query: np.ndarray, reference: np.ndarray) -> float:
    """Normalized cross-correlation between query and reference fingerprint."""
    w = normalized_residual(query)
    h = min(w.shape[0], reference.shape[0])
    ww = min(w.shape[1], reference.shape[1])
    w_crop = w[:h, :ww].astype(np.float64).ravel()
    k_crop = reference[:h, :ww].astype(np.float64).ravel()
    w_norm = w_crop - w_crop.mean()
    k_norm = k_crop - k_crop.mean()
    denom = np.linalg.norm(w_norm) * np.linalg.norm(k_norm)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(w_norm, k_norm) / denom)


def analyze_prnu(
    img: np.ndarray,
    reference_fingerprint: np.ndarray | None = None,
) -> dict[str, Any]:
    if reference_fingerprint is None:
        return {
            "status": "reference_fingerprint_unavailable",
            "prnu_reference_available": False,
            "prnu_supported": True,
            "prnu_similarity": None,
            "prnu_quality": None,
            "availability": False,
        }

    w = normalized_residual(img)
    quality = float(np.std(w))
    similarity = prnu_correlation(img, reference_fingerprint)

    return {
        "status": "success",
        "prnu_reference_available": True,
        "prnu_supported": True,
        "prnu_similarity": similarity,
        "prnu_quality": quality,
        "availability": True,
    }


def save_fingerprint(fingerprint: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, fingerprint)


def load_fingerprint(path: Path) -> np.ndarray:
    return np.load(path)
