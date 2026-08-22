"""Tests for metadata extraction."""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, PngImagePlugin

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


def test_extracts_png_text_metadata(tmp_path):
    path = tmp_path / "labelled.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "Example editor")
    info.add_text("CameraModel", "Demo Phone")
    Image.new("RGB", (20, 20)).save(path, pnginfo=info)

    meta = extract_metadata(path)

    assert meta["metadata_available"] is True
    assert "CONTAINER" in meta["metadata_sources"]
    assert meta["metadata_camera_model"] == "Demo Phone"


def test_extracts_xmp_camera_metadata(tmp_path):
    path = tmp_path / "xmp.jpg"
    Image.new("RGB", (20, 20)).save(path)
    path.write_bytes(path.read_bytes() + b'''\n<x:xmpmeta xmlns:x="adobe:ns:meta/">
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Make="vivo" tiff:Model="vivo T2x 5G" />
</rdf:RDF></x:xmpmeta>''')

    meta = extract_metadata(path)

    assert "XMP" in meta["metadata_sources"]
    assert meta["metadata_camera_make"] == "vivo"
    assert meta["metadata_camera_model"] == "vivo T2x 5G"
