"""Tests for manipulation detection."""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from app.forensic.manipulation.copy_move import detect_copy_move
from app.forensic.manipulation.noise_inconsistency import local_noise_inconsistency
from app.forensic.manipulation.ela import analyze_ela


def test_local_noise_inconsistency():
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    result = local_noise_inconsistency(img)
    assert result["status"] == "success"
    assert "local_noise_anomaly_score" in result


def test_copy_move_no_detection_on_random():
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    result = detect_copy_move(img)
    assert "copy_move_detected" in result


def test_ela_on_jpeg():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.jpg"
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)
        result = analyze_ela(path)
        assert result["availability"] is True
