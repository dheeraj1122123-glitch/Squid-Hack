"""Metadata vs signal consistency checks."""
from typing import Any


def check_metadata_signal_mismatch(
    metadata: dict[str, Any],
    predicted_make: str | None,
    predicted_model: str | None,
) -> dict[str, Any]:
    meta_make = (metadata.get("metadata_camera_make") or "").lower().strip()
    meta_model = (metadata.get("metadata_camera_model") or "").lower().strip()
    pred_make = (predicted_make or "").lower().strip()
    pred_model = (predicted_model or "").lower().strip()

    mismatch = False
    reasons: list[str] = []

    if meta_make and pred_make and meta_make not in pred_make and pred_make not in meta_make:
        mismatch = True
        reasons.append("manufacturer_mismatch")

    if meta_model and pred_model:
        meta_tokens = set(meta_model.replace("/", " ").split())
        pred_tokens = set(pred_model.replace("/", " ").split())
        overlap = meta_tokens & pred_tokens
        if not overlap and meta_model not in pred_model and pred_model not in meta_model:
            mismatch = True
            reasons.append("model_mismatch")

    return {
        "metadata_signal_mismatch": mismatch,
        "mismatch_reasons": reasons,
    }
