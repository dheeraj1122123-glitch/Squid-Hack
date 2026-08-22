"""Splicing detection baseline."""
from typing import Any

import numpy as np

from app.forensic.manipulation.jpeg_analysis import analyze_jpeg_manipulation
from app.forensic.manipulation.noise_inconsistency import local_noise_inconsistency
from app.forensic.preprocessing.patching import extract_patches


def detect_splicing(img: np.ndarray, path=None) -> dict[str, Any]:
    noise_result = local_noise_inconsistency(img)
    jpeg_result = analyze_jpeg_manipulation(img, path)

    patch_scores: list[float] = []
    patch_size = 128
    for patch, (y, x) in extract_patches(img, patch_size, patch_size // 2):
        n = local_noise_inconsistency(patch, patch_size=32, stride=16)
        patch_scores.append(n.get("local_noise_anomaly_score", 0.0))

    patch_array = np.array(patch_scores) if patch_scores else np.array([0.0])
    tampered_prob = float(
        0.4 * noise_result.get("local_noise_anomaly_score", 0)
        + 0.3 * jpeg_result.get("compression_inconsistency_score", 0)
        + 0.3 * np.mean(patch_array[patch_array > np.percentile(patch_array, 75)])
    )
    tampered_prob = min(1.0, max(0.0, tampered_prob))

    return {
        "status": "success",
        "availability": True,
        "tampered_probability": tampered_prob,
        "patch_scores": patch_scores,
        "noise_evidence": noise_result,
        "jpeg_evidence": jpeg_result,
        "limitations": ["Patch-based baseline; not state-of-the-art"],
    }
