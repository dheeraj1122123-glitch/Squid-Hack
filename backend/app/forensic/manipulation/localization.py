"""Unified suspicious region localization."""
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def fuse_localization_maps(
    h: int,
    w: int,
    ela_result: dict | None = None,
    noise_result: dict | None = None,
    copy_move_result: dict | None = None,
    splice_result: dict | None = None,
    jpeg_result: dict | None = None,
) -> dict[str, Any]:
    combined = np.zeros((h, w), dtype=np.float32)
    weights: list[tuple[np.ndarray, float]] = []

    if noise_result and noise_result.get("anomaly_map") is not None:
        amap = noise_result["anomaly_map"]
        if amap.shape != (h, w):
            amap = cv2.resize(amap, (w, h))
        norm = amap / (amap.max() + 1e-8)
        weights.append((norm, 0.3))

    if copy_move_result and copy_move_result.get("heatmap") is not None:
        hm = copy_move_result["heatmap"]
        if hm.shape != (h, w):
            hm = cv2.resize(hm, (w, h))
        weights.append((hm, 0.25))

    if ela_result and ela_result.get("ela_map") is not None:
        ela = ela_result["ela_map"]
        if ela.ndim == 3:
            ela = np.mean(ela, axis=2)
        if ela.shape != (h, w):
            ela = cv2.resize(ela, (w, h))
        norm = ela / (ela.max() + 1e-8)
        weights.append((norm, 0.2))

    jpeg_score = jpeg_result.get("compression_inconsistency_score", 0) if jpeg_result else 0
    splice_score = splice_result.get("tampered_probability", 0) if splice_result else 0

    total_weight = sum(w for _, w in weights)
    if total_weight > 0:
        for m, wt in weights:
            combined += m * (wt / total_weight)

    combined += jpeg_score * 0.1
    combined += splice_score * 0.15
    combined = np.clip(combined, 0, 1)

    regions = _extract_regions(combined)
    overall = float(np.mean(combined[combined > np.percentile(combined, 90)])) if combined.max() > 0 else 0.0

    return {
        "heatmap": combined,
        "overall_suspiciousness": min(1.0, overall),
        "regions": regions,
        "modules_used": len(weights),
    }


def _extract_regions(heatmap: np.ndarray, threshold: float = 0.5) -> list[dict]:
    binary = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < 100:
            continue
        mask = np.zeros_like(heatmap)
        cv2.drawContours(mask, [cnt], -1, 1.0, -1)
        score = float(np.mean(heatmap[mask > 0]))
        regions.append({"x": x, "y": y, "width": bw, "height": bh, "score": score})
    regions.sort(key=lambda r: r["score"], reverse=True)
    return regions[:10]


def save_heatmap(heatmap: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vis = (heatmap * 255).astype(np.uint8)
    cv2.imwrite(str(path), vis)
