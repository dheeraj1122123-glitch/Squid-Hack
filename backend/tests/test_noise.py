"""Tests for noise extraction."""
import numpy as np

from app.forensic.camera.noise import (
    extract_noise_residual,
    neighbor_correlation,
    residual_statistics,
)


def test_noise_residual_shape():
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    residual = extract_noise_residual(img)
    assert residual.shape == img.shape


def test_residual_statistics():
    residual = np.random.randn(32, 32).astype(np.float64) * 0.01
    stats = residual_statistics(residual)
    assert "mean" in stats
    assert "std" in stats
    assert "skewness" in stats
    assert "kurtosis" in stats


def test_neighbor_correlation():
    residual = np.random.randn(32, 32).astype(np.float64) * 0.01
    corr = neighbor_correlation(residual)
    assert -1 <= corr <= 1
