"""Color space conversions."""
import cv2
import numpy as np


def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        u8 = img
    return cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        u8 = img
    return cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)


def channel_correlation(img: np.ndarray) -> dict[str, float]:
    if img.ndim != 3 or img.shape[2] < 3:
        return {"rg": 0.0, "rb": 0.0, "gb": 0.0}
    r = img[:, :, 0].astype(np.float64).ravel()
    g = img[:, :, 1].astype(np.float64).ravel()
    b = img[:, :, 2].astype(np.float64).ravel()
    return {
        "rg": float(np.corrcoef(r, g)[0, 1]),
        "rb": float(np.corrcoef(r, b)[0, 1]),
        "gb": float(np.corrcoef(g, b)[0, 1]),
    }
