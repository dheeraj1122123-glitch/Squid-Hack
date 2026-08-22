"""JPEG manipulation inconsistency analysis."""
from pathlib import Path
from typing import Any

from app.forensic.camera.jpeg import analyze_compression


def analyze_jpeg_manipulation(img, path: Path | None = None) -> dict[str, Any]:
    result = analyze_compression(img, path)
    score = 0.0
    if result.get("possible_recompression"):
        score += 0.4
    quality = result.get("compression_quality_estimate")
    if quality is not None and quality < 70:
        score += 0.2
    return {
        "status": "success",
        "availability": True,
        "compression_inconsistency_score": min(1.0, score),
        "compression_features": result,
        "limitations": ["JPEG analysis alone cannot prove tampering"],
    }
