"""CFA / demosaicing related feature extraction."""
from typing import Any

import numpy as np
from scipy import ndimage

from app.forensic.preprocessing.color import channel_correlation
from app.forensic.preprocessing.normalization import to_float01


def local_channel_correlations(img: np.ndarray, block: int = 32) -> dict[str, float]:
    f = to_float01(img)
    if f.ndim != 3 or f.shape[2] < 3:
        return {"local_rg": 0.0, "local_rb": 0.0, "local_gb": 0.0}
    h, w = f.shape[:2]
    corrs = {"rg": [], "rb": [], "gb": []}
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            patch = f[y : y + block, x : x + block]
            c = channel_correlation((patch * 255).astype(np.uint8))
            corrs["rg"].append(c["rg"])
            corrs["rb"].append(c["rb"])
            corrs["gb"].append(c["gb"])
    return {
        "local_rg": float(np.mean(corrs["rg"])) if corrs["rg"] else 0.0,
        "local_rb": float(np.mean(corrs["rb"])) if corrs["rb"] else 0.0,
        "local_gb": float(np.mean(corrs["gb"])) if corrs["gb"] else 0.0,
    }


def periodic_2x2_stats(img: np.ndarray) -> dict[str, float]:
    f = to_float01(img)
    if f.ndim != 3:
        return {"periodic_energy": 0.0}
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    even_even = r[0::2, 0::2].mean()
    even_odd = g[0::2, 1::2].mean()
    odd_even = g[1::2, 0::2].mean()
    odd_odd = b[1::2, 1::2].mean()
    vals = np.array([even_even, even_odd, odd_even, odd_odd])
    return {
        "periodic_mean": float(vals.mean()),
        "periodic_std": float(vals.std()),
        "periodic_energy": float(np.sum(vals ** 2)),
        "rg_diff_even": float(abs(even_even - even_odd)),
        "gb_diff_odd": float(abs(odd_even - odd_odd)),
    }


def directional_correlations(img: np.ndarray) -> dict[str, float]:
    f = to_float01(img)
    if f.ndim == 3:
        gray = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
    else:
        gray = f
    h = np.corrcoef(gray[:, :-1].ravel(), gray[:, 1:].ravel())[0, 1]
    v = np.corrcoef(gray[:-1, :].ravel(), gray[1:, :].ravel())[0, 1]
    d1 = np.corrcoef(gray[:-1, :-1].ravel(), gray[1:, 1:].ravel())[0, 1]
    d2 = np.corrcoef(gray[:-1, 1:].ravel(), gray[1:, :-1].ravel())[0, 1]
    return {"dir_h": float(h), "dir_v": float(v), "dir_d1": float(d1), "dir_d2": float(d2)}


def inter_channel_differences(img: np.ndarray) -> dict[str, float]:
    f = to_float01(img)
    if f.ndim != 3:
        return {}
    rg = f[:, :, 0] - f[:, :, 1]
    rb = f[:, :, 0] - f[:, :, 2]
    gb = f[:, :, 1] - f[:, :, 2]
    return {
        "rg_diff_mean": float(np.mean(np.abs(rg))),
        "rb_diff_mean": float(np.mean(np.abs(rb))),
        "gb_diff_mean": float(np.mean(np.abs(gb))),
        "rg_diff_std": float(np.std(rg)),
        "rb_diff_std": float(np.std(rb)),
        "gb_diff_std": float(np.std(gb)),
    }


def extract_cfa_features(img: np.ndarray) -> dict[str, Any]:
    global_corr = channel_correlation(img if img.dtype == np.uint8 else (np.clip(img, 0, 1) * 255).astype(np.uint8))
    return {
        "status": "success",
        "availability": True,
        "demosaicing_related_features": {
            "global_channel_correlation": global_corr,
            "local_channel_correlation": local_channel_correlations(img),
            "periodic_2x2": periodic_2x2_stats(img),
            "directional_correlation": directional_correlations(img),
            "inter_channel_differences": inter_channel_differences(img),
        },
    }
