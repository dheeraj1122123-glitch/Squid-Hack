"""AI-generated image detector module (Synthetic vs Real).

Operates independently from the camera attribution model.
"""
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from app.core.config import settings
from app.forensic.preprocessing.normalization import to_float01


class AIGeneratedClassifier(nn.Module):
    """ResNet-like feature extractor for synthetic image detection."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AIGeneratorDetector:
    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or settings.model_dir / "manipulation" / "ai_generator_detector.pt"
        self.model: AIGeneratedClassifier | None = None
        self.trained = False
        self._load()

    def _load(self) -> None:
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location="cpu")
                self.model = AIGeneratedClassifier()
                self.model.load_state_dict(checkpoint["state_dict"])
                self.model.eval()
                self.trained = True
            except Exception:
                self.trained = False

    def predict(self, img: np.ndarray) -> dict[str, Any]:
        if not settings.enable_ai_detector or not self.trained or self.model is None:
            return {
                "status": "model_unavailable",
                "supported": True,
                "ai_generated_likelihood": 0.0,
                "model_version": "1.0.0",
                "message": "AI detector model not trained / unavailable",
            }

        f = to_float01(img)
        if f.ndim == 2:
            f = np.stack([f] * 3, axis=-1)

        # Resize to standard size 256x256
        import cv2
        resized = cv2.resize(f, (256, 256))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float()

        with torch.no_grad():
            prob = float(self.model(tensor).item())

        return {
            "status": "success",
            "supported": True,
            "ai_generated_likelihood": prob,
            "is_ai_generated": prob > 0.5,
            "model_version": "1.0.0",
        }
