"""Image normalization utilities."""
import numpy as np


def to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64)


def normalize_channels(img: np.ndarray) -> np.ndarray:
    f = to_float01(img)
    if f.ndim == 2:
        return f
    out = np.zeros_like(f)
    for c in range(f.shape[2]):
        ch = f[:, :, c]
        mn, mx = ch.min(), ch.max()
        out[:, :, c] = (ch - mn) / (mx - mn + 1e-8)
    return out


def per_channel_zscore(img: np.ndarray) -> np.ndarray:
    f = to_float01(img)
    if f.ndim == 2:
        return (f - f.mean()) / (f.std() + 1e-8)
    out = np.zeros_like(f)
    for c in range(f.shape[2]):
        ch = f[:, :, c]
        out[:, :, c] = (ch - ch.mean()) / (ch.std() + 1e-8)
    return out
