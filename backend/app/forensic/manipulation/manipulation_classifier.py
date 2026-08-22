"""Learned manipulation classifier interface."""
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.core.config import settings


class ManipulationClassifier:
    MANIPULATION_TYPES = [
        "AUTHENTIC",
        "COPY_MOVE",
        "SPLICING",
        "OBJECT_REMOVAL_OR_INPAINTING",
        "RETOUCHING",
        "RECOMPRESSION_ONLY",
        "UNKNOWN_MANIPULATION",
    ]

    def __init__(self):
        self.model = None
        self.trained = False
        self._load()

    def _load(self) -> None:
        registry_path = settings.model_dir / "registry.json"
        if registry_path.exists():
            registry = json.loads(registry_path.read_text())
            info = registry.get("manipulation_detector", {})
            self.trained = info.get("trained", False)

        model_path = settings.model_dir / "manipulation" / "manipulation_detector.joblib"
        if model_path.exists():
            self.model = joblib.load(model_path)
            self.trained = True

    def predict(self, features: np.ndarray) -> dict[str, Any]:
        if not self.trained or self.model is None:
            return {
                "status": "MODEL_NOT_TRAINED",
                "availability": False,
                "predicted_type": None,
            }

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features.reshape(1, -1))[0]
            classes = self.model.classes_
            best = int(np.argmax(proba))
            return {
                "status": "success",
                "availability": True,
                "predicted_type": str(classes[best]),
                "confidence": float(proba[best]),
                "probabilities": {str(c): float(p) for c, p in zip(classes, proba)},
            }

        pred = self.model.predict(features.reshape(1, -1))[0]
        return {
            "status": "success",
            "availability": True,
            "predicted_type": str(pred),
            "confidence": 0.5,
        }
