"""Deep residual patch model for camera model identification.

Uses noise residual patches as input to prevent semantic shortcut learning.
"""
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from app.core.config import settings
from app.forensic.camera.noise import extract_noise_residual
from app.forensic.preprocessing.patching import extract_patches


class CameraResidualCNN(nn.Module):
    """Lightweight CNN operating on noise residual patches."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class DeepResidualPredictor:
    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or settings.model_dir / "camera" / "camera_residual_model.pt"
        self.model: CameraResidualCNN | None = None
        self.classes: list[str] = []
        self.trained = False
        self._load()

    def _load(self) -> None:
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location="cpu")
                self.classes = checkpoint.get("classes", [])
                num_classes = len(self.classes) if self.classes else 10
                self.model = CameraResidualCNN(num_classes=num_classes)
                self.model.load_state_dict(checkpoint["state_dict"])
                self.model.eval()
                self.trained = True
            except Exception:
                self.trained = False

    def predict(self, img: np.ndarray, patch_size: int = 128) -> dict[str, Any]:
        if not self.trained or self.model is None:
            return {
                "status": "model_unavailable",
                "trained": False,
                "camera_model": None,
                "confidence": 0.0,
                "message": "Deep residual model not trained / unavailable",
            }

        residual = extract_noise_residual(img, method="gaussian")
        patches = []
        for p, _ in extract_patches(residual, patch_size=patch_size, stride=patch_size):
            if p.ndim == 2:
                p = np.stack([p] * 3, axis=-1)
            # convert (H, W, C) to (C, H, W)
            p_tensor = torch.from_numpy(p.transpose(2, 0, 1)).float()
            patches.append(p_tensor)

        if not patches:
            return {
                "status": "insufficient_signal",
                "trained": True,
                "camera_model": None,
                "confidence": 0.0,
            }

        batch = torch.stack(patches)
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1).mean(dim=0).numpy()

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        pred_class = self.classes[best_idx] if best_idx < len(self.classes) else f"Class_{best_idx}"

        return {
            "status": "success",
            "trained": True,
            "camera_model": pred_class,
            "confidence": confidence,
            "all_probabilities": {
                self.classes[i] if i < len(self.classes) else str(i): float(probs[i])
                for i in range(len(probs))
            },
        }
