"""Camera forensic feature vector construction."""
from typing import Any

import numpy as np


def flatten_dict(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}_{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        elif isinstance(v, (int, float, np.floating, np.integer)):
            out[key] = float(v)
        elif v is None:
            out[key] = 0.0
    return out


def build_feature_vector(
    noise: dict[str, Any],
    cfa: dict[str, Any],
    frequency: dict[str, Any],
    compression: dict[str, Any],
    prnu: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[str]]:
    features: dict[str, float] = {}

    if noise.get("combined_statistics"):
        features.update(flatten_dict(noise["combined_statistics"], "noise"))

    if cfa.get("demosaicing_related_features"):
        features.update(flatten_dict(cfa["demosaicing_related_features"], "cfa"))

    for key in (
        "low_frequency_energy",
        "mid_frequency_energy",
        "high_frequency_energy",
        "spectral_entropy",
        "radial_profile_mean",
        "radial_profile_std",
    ):
        if key in frequency:
            features[f"freq_{key}"] = float(frequency[key])

    if compression.get("compression_features"):
        features.update(flatten_dict(compression["compression_features"], "jpeg"))
    if compression.get("compression_quality_estimate") is not None:
        features["jpeg_quality"] = float(compression["compression_quality_estimate"] or 0)
    features["jpeg_possible_recompression"] = float(
        1.0 if compression.get("possible_recompression") else 0.0
    )

    if prnu and prnu.get("prnu_similarity") is not None:
        features["prnu_similarity"] = float(prnu["prnu_similarity"])
        features["prnu_quality"] = float(prnu.get("prnu_quality") or 0)
    else:
        features["prnu_similarity"] = 0.0
        features["prnu_quality"] = 0.0

    names = sorted(features.keys())
    vector = np.array([features[n] for n in names], dtype=np.float64)
    return vector, names
