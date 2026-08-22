"""Image loading utilities."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.core.exceptions import ImageValidationError


def load_image_bgr(path: Path | str) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise ImageValidationError(f"File not found: {path}")
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        img = None

    if img is None:
        try:
            with Image.open(path) as im:
                rgb = np.array(im.convert("RGB"))
                img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ImageValidationError(f"Cannot decode image: {path} ({e})")
    return img


def load_image_rgb(path: Path | str) -> np.ndarray:
    bgr = load_image_bgr(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_image_gray(path: Path | str) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise ImageValidationError(f"File not found: {path}")
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        img = None

    if img is None:
        try:
            with Image.open(path) as im:
                img = np.array(im.convert("L"))
        except Exception as e:
            raise ImageValidationError(f"Cannot decode grayscale image: {path} ({e})")
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
