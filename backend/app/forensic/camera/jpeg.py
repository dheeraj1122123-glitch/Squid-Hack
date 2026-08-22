"""JPEG compression analysis."""
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.forensic.preprocessing.normalization import to_float01


def estimate_jpeg_quality(path: Path) -> int | None:
    try:
        with Image.open(path) as im:
            if im.format != "JPEG":
                return None
            qtables = im.quantization
            if not qtables:
                return None
            table = qtables.get(0) or next(iter(qtables.values()))
            avg_q = sum(table) / len(table)
            quality = max(1, min(100, int(100 - avg_q / 2.55)))
            return quality
    except Exception:
        return None


def dct_statistics(img: np.ndarray) -> dict[str, float]:
    f = to_float01(img)
    if f.ndim == 3:
        gray = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (f * 255).astype(np.uint8)
    gray = np.float32(gray)
    dct = cv2.dct(gray)
    abs_dct = np.abs(dct)
    return {
        "dct_mean": float(np.mean(abs_dct)),
        "dct_std": float(np.std(abs_dct)),
        "dct_energy_low": float(np.sum(abs_dct[:8, :8] ** 2)),
        "dct_energy_high": float(np.sum(abs_dct[8:, 8:] ** 2)),
    }


def double_compression_indicator(img: np.ndarray, path: Path | None = None) -> dict[str, Any]:
    """Heuristic double-compression indicator via re-save difference."""
    if path and path.suffix.lower() in (".jpg", ".jpeg"):
        try:
            with open(path, "rb") as f:
                original_bytes = f.read()
            im = Image.open(BytesIO(original_bytes)).convert("RGB")
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=90)
            resaved = np.array(Image.open(BytesIO(buf.getvalue())))
            orig = np.array(im)
            diff = np.abs(orig.astype(float) - resaved.astype(float)).mean()
            return {
                "possible_recompression": diff > 2.0,
                "resave_difference_mean": float(diff),
            }
        except Exception:
            pass
    return {"possible_recompression": False, "resave_difference_mean": 0.0}


def analyze_compression(img: np.ndarray, path: Path | None = None) -> dict[str, Any]:
    quality = estimate_jpeg_quality(path) if path else None
    dct_stats = dct_statistics(img)
    recomp = double_compression_indicator(img, path)

    return {
        "status": "success",
        "availability": True,
        "compression_features": dct_stats,
        "compression_quality_estimate": quality,
        "possible_recompression": recomp["possible_recompression"],
        "resave_difference_mean": recomp["resave_difference_mean"],
    }
