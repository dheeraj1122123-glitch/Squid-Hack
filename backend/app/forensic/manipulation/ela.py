"""Error Level Analysis."""
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def compute_ela(img_path: Path, quality: int = 90) -> tuple[np.ndarray, dict[str, float]]:
    with open(img_path, "rb") as f:
        original_bytes = f.read()

    try:
        im = Image.open(BytesIO(original_bytes)).convert("RGB")
    except Exception:
        im = Image.open(img_path).convert("RGB")

    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    resaved = np.array(Image.open(BytesIO(buf.getvalue())))
    original = np.array(im)

    ela = np.abs(original.astype(np.float32) - resaved.astype(np.float32))
    ela_gray = np.mean(ela, axis=2)

    stats = {
        "ela_mean": float(np.mean(ela_gray)),
        "ela_std": float(np.std(ela_gray)),
        "ela_max": float(np.max(ela_gray)),
        "ela_energy": float(np.sum(ela_gray ** 2) / ela_gray.size),
        "ela_high_ratio": float(np.mean(ela_gray > ela_gray.mean() + 2 * ela_gray.std())),
    }

    return ela, stats


def analyze_ela(img_path: Path, artifact_dir: Path | None = None) -> dict[str, Any]:
    is_jpeg = img_path.suffix.lower() in (".jpg", ".jpeg")
    if not is_jpeg:
        return {
            "status": "skipped",
            "availability": False,
            "reason": "ELA requires JPEG input",
            "statistics": {},
        }

    ela, stats = compute_ela(img_path)
    artifact_path = None
    if artifact_dir:
        artifact_path = artifact_dir / "ela.png"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        vis = ela / (ela.max() + 1e-8) * 255
        cv2.imwrite(str(artifact_path), cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2BGR))

    return {
        "status": "success",
        "availability": True,
        "statistics": stats,
        "artifact_path": str(artifact_path) if artifact_path else None,
        "ela_map": ela,
        "limitations": ["ELA alone is not proof of manipulation"],
    }
