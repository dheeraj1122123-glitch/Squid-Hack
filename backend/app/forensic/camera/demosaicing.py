"""Demosaicing artifact analysis (image-level features)."""
from app.forensic.camera.cfa import extract_cfa_features


def analyze_demosaicing(img) -> dict:
    return extract_cfa_features(img)
