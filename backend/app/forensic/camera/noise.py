"""Noise residual extraction and statistics."""
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage, stats
from scipy.ndimage import median_filter

from app.forensic.preprocessing.normalization import to_float01


def gaussian_denoise(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    f = to_float01(img)
    if f.ndim == 2:
        return ndimage.gaussian_filter(f, sigma=sigma)
    return np.stack([ndimage.gaussian_filter(f[:, :, c], sigma=sigma) for c in range(3)], axis=-1)


def median_denoise(img: np.ndarray, size: int = 3) -> np.ndarray:
    f = to_float01(img)
    if f.ndim == 2:
        return median_filter(f, size=size)
    return np.stack([median_filter(f[:, :, c], size=size) for c in range(3)], axis=-1)


def highpass_residual(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    f = to_float01(img)
    if f.ndim == 2:
        blur = cv2.GaussianBlur(f, (ksize, ksize), 0)
        return f - blur
    out = np.zeros_like(f)
    for c in range(3):
        blur = cv2.GaussianBlur(f[:, :, c], (ksize, ksize), 0)
        out[:, :, c] = f[:, :, c] - blur
    return out


def wavelet_residual(img: np.ndarray) -> np.ndarray:
    """Extract wavelet high-frequency noise residual using 2D Haar wavelets."""
    f = to_float01(img)
    kernel_h = np.array([[-0.5, 0.5]], dtype=np.float32)
    kernel_v = np.array([[-0.5], [0.5]], dtype=np.float32)
    kernel_d = np.array([[0.25, -0.25], [-0.25, 0.25]], dtype=np.float32)
    
    if f.ndim == 2:
        lh = ndimage.convolve(f, kernel_h)
        hl = ndimage.convolve(f, kernel_v)
        hh = ndimage.convolve(f, kernel_d)
        return (lh + hl + hh) / 3.0
    
    out = np.zeros_like(f)
    for c in range(f.shape[2]):
        ch = f[:, :, c]
        lh = ndimage.convolve(ch, kernel_h)
        hl = ndimage.convolve(ch, kernel_v)
        hh = ndimage.convolve(ch, kernel_d)
        out[:, :, c] = (lh + hl + hh) / 3.0
    return out


def extract_noise_residual(img: np.ndarray, method: str = "gaussian") -> np.ndarray:
    f = to_float01(img)
    if method == "gaussian":
        denoised = gaussian_denoise(img)
        return f - denoised
    elif method == "median":
        denoised = median_denoise(img)
        return f - denoised
    elif method == "highpass":
        return highpass_residual(img)
    elif method == "wavelet":
        return wavelet_residual(img)
    else:
        denoised = gaussian_denoise(img)
        return f - denoised


def residual_statistics(residual: np.ndarray) -> dict[str, float]:
    flat = residual.ravel().astype(np.float64)
    local_var = float(np.var(residual))
    energy = float(np.sum(residual ** 2) / max(1, residual.size))
    result = {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "variance": float(np.var(flat)),
        "skewness": float(stats.skew(flat)),
        "kurtosis": float(stats.kurtosis(flat)),
        "energy": energy,
        "local_variance": local_var,
    }
    if residual.ndim == 3 and residual.shape[2] >= 3:
        ch = [residual[:, :, c].ravel() for c in range(3)]
        result["channel_corr_rg"] = float(np.corrcoef(ch[0], ch[1])[0, 1])
        result["channel_corr_rb"] = float(np.corrcoef(ch[0], ch[2])[0, 1])
        result["channel_corr_gb"] = float(np.corrcoef(ch[1], ch[2])[0, 1])
    return result


def neighbor_correlation(residual: np.ndarray) -> float:
    if residual.ndim == 3:
        r = residual[:, :, 0]
    else:
        r = residual
    h_corr = np.corrcoef(r[:, :-1].ravel(), r[:, 1:].ravel())[0, 1]
    v_corr = np.corrcoef(r[:-1, :].ravel(), r[1:, :].ravel())[0, 1]
    return float((h_corr + v_corr) / 2)


def save_residual_visualization(residual: np.ndarray, path: Path) -> None:
    if residual.ndim == 3:
        mag = np.sqrt(np.sum(residual ** 2, axis=2))
    else:
        mag = np.abs(residual)
    mag = mag - mag.min()
    if mag.max() > 0:
        mag = mag / mag.max()
    vis = (mag * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


def analyze_noise(img: np.ndarray, artifact_dir: Path | None = None) -> dict[str, Any]:
    residual = extract_noise_residual(img, method="gaussian")
    per_channel: dict[str, dict] = {}
    if img.ndim == 3 and img.shape[2] >= 3:
        for i, name in enumerate(["R", "G", "B"]):
            ch_res = extract_noise_residual(img[:, :, i], method="gaussian")
            per_channel[name] = residual_statistics(ch_res if ch_res.ndim == 2 else ch_res[:, :, 0])
    else:
        per_channel["combined"] = residual_statistics(residual)

    stats_combined = residual_statistics(residual)
    stats_combined["neighbor_correlation"] = neighbor_correlation(residual)

    artifact_path = None
    if artifact_dir:
        artifact_path = artifact_dir / "noise_residual.png"
        save_residual_visualization(residual, artifact_path)

    return {
        "status": "success",
        "availability": True,
        "method": "gaussian_smoothing_residual",
        "combined_statistics": stats_combined,
        "channel_statistics": per_channel,
        "artifact_path": str(artifact_path) if artifact_path else None,
        "residual": residual,
    }
