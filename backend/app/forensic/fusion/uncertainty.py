"""Uncertainty quantification."""
from typing import Any


def compute_uncertainty(
    confidence: float | None,
    module_availability: dict[str, bool],
    min_modules: int = 3,
) -> dict[str, Any]:
    available_count = sum(1 for v in module_availability.values() if v)
    module_factor = 1.0 - (available_count / max(len(module_availability), 1))

    if confidence is None:
        return {
            "confidence": 0.0,
            "uncertainty": 1.0,
            "status": "inconclusive",
            "reason": "insufficient calibrated confidence",
        }

    uncertainty = (1.0 - confidence) * 0.7 + module_factor * 0.3
    status = "inconclusive" if uncertainty > 0.6 or available_count < min_modules else "confident"

    return {
        "confidence": float(confidence),
        "uncertainty": float(min(1.0, uncertainty)),
        "status": status,
        "modules_available": available_count,
    }
