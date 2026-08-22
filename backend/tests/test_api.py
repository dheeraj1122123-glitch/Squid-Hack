"""Integration tests for API."""
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture
def test_image():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        cv2.imwrite(f.name, img)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


def test_upload_and_analyze(test_image):
    with open(test_image, "rb") as f:
        response = client.post(
            "/api/v1/analysis/upload",
            files={"file": ("test.jpg", f, "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "analysis_id" in data
    assert "sha256" in data
    analysis_id = data["analysis_id"]

    response = client.post(f"/api/v1/analysis/{analysis_id}/run?sync=true")
    assert response.status_code == 200

    result = client.get(f"/api/v1/analysis/{analysis_id}").json()
    assert result["status"] in ("COMPLETED", "FAILED", "REPORT_GENERATION")

    if result["status"] == "COMPLETED":
        evidence = client.get(f"/api/v1/analysis/{analysis_id}/evidence")
        assert evidence.status_code == 200

        camera = client.get(f"/api/v1/analysis/{analysis_id}/camera")
        assert camera.status_code == 200
        cam_data = camera.json()
        assert cam_data["status"] in ("MODEL_NOT_TRAINED", "success", "UNKNOWN_CAMERA")

        manip = client.get(f"/api/v1/analysis/{analysis_id}/manipulation")
        assert manip.status_code == 200

        report = client.get(f"/api/v1/analysis/{analysis_id}/report")
        assert report.status_code == 200


def test_invalid_upload():
    response = client.post(
        "/api/v1/analysis/upload",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
