"""Image loading utilities."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.core.exceptions import ImageValidationError


def load_image_bgr(path: Path | str) -> np.ndarray:
    path = Path(path)
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageValidationError(f"Cannot decode image: {path}")
    return img


def load_image_rgb(path: Path | str) -> np.ndarray:
    bgr = load_image_bgr(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_image_gray(path: Path | str) -> np.ndarray:
    path = Path(path)
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ImageValidationError(f"Cannot decode grayscale image: {path}")
    return img


def get_image_info(path: Path) -> dict:
    with Image.open(path) as im:
        return {
            "width": im.width,
            "height": im.height,
            "channels": len(im.getbands()),
            "format": im.format or "UNKNOWN",
            "mode": im.mode,
        }


def bytes_to_rgb(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ImageValidationError("Cannot decode image from bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
