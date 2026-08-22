"""Tests for metadata extraction."""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.forensic.metadata.extractor import extract_metadata


@pytest.fixture
def sample_image():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.jpg"
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)
        yield path


def test_extract_metadata_no_exif(sample_image):
    meta = extract_metadata(sample_image)
    assert "metadata_available" in meta
    assert "metadata_suspicious_flags" in meta
    assert isinstance(meta["metadata_suspicious_flags"], list)
