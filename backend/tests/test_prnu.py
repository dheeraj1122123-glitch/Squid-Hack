"""Tests for PRNU pipeline."""
import numpy as np

from app.forensic.camera.prnu import aggregate_fingerprint, normalized_residual, prnu_correlation


def test_normalized_residual():
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    w = normalized_residual(img)
    assert w.shape == img.shape
    assert np.std(w) > 0


def test_prnu_correlation_same_camera():
    rng = np.random.RandomState(42)
    base = rng.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    noise = rng.normal(0, 1, base.shape)
    img1 = np.clip(base.astype(float) + noise, 0, 255).astype(np.uint8)
    img2 = np.clip(base.astype(float) + rng.normal(0, 1, base.shape), 0, 255).astype(np.uint8)

    ref = aggregate_fingerprint([img1, img2])
    sim = prnu_correlation(img1, ref)
    assert -1 <= sim <= 1


def test_prnu_unavailable_without_reference():
    from app.forensic.camera.prnu import analyze_prnu
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = analyze_prnu(img, reference_fingerprint=None)
    assert result["status"] == "reference_fingerprint_unavailable"
