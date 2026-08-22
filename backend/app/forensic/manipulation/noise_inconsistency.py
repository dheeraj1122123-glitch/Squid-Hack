"""Local noise inconsistency detection."""
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.forensic.camera.noise import extract_noise_residual, residual_statistics
from app.forensic.preprocessing.patching import extract_patches


def local_noise_inconsistency(img: np.ndarray, patch_size: int = 64, stride: int = 32) -> dict[str, Any]:
    residual = extract_noise_residual(img)
    if residual.ndim == 3:
        res_gray = np.sqrt(np.sum(residual ** 2, axis=2))
    else:
        res_gray = np.abs(residual)

    h, w = res_gray.shape
    anomaly_map = np.zeros((h, w), dtype=np.float32)
    patch_stats: list[dict] = []

    for patch, (y, x) in extract_patches(res_gray, patch_size, stride):
        stats = {
            "variance": float(np.var(patch)),
            "energy": float(np.sum(patch ** 2) / patch.size),
            "mean": float(np.mean(patch)),
        }
        patch_stats.append({**stats, "y": y, "x": x})

    if not patch_stats:
        return {
            "status": "insufficient_signal",
            "local_noise_anomaly_score": 0.0,
            "availability": False,
        }

    variances = np.array([p["variance"] for p in patch_stats])
    global_mean = variances.mean()
    global_std = variances.std() + 1e-8

    for p in patch_stats:
        z = abs(p["variance"] - global_mean) / global_std
        y, x = p["y"], p["x"]
        anomaly_map[y : y + patch_size, x : x + patch_size] = np.maximum(
            anomaly_map[y : y + patch_size, x : x + patch_size], z
        )

    anomaly_score = float(np.mean(anomaly_map[anomaly_map > 1.5])) if (anomaly_map > 1.5).any() else 0.0
    if np.isnan(anomaly_score):
        anomaly_score = float(np.max(anomaly_map))

    return {
        "status": "success",
        "availability": True,
        "local_noise_anomaly_score": min(1.0, anomaly_score / 5.0),
        "anomaly_map": anomaly_map,
        "patch_count": len(patch_stats),
    }


def save_noise_heatmap(anomaly_map: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    norm = anomaly_map - anomaly_map.min()
    if norm.max() > 0:
        norm = norm / norm.max()
    cv2.imwrite(str(path), (norm * 255).astype(np.uint8))
