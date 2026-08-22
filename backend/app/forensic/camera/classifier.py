"""Camera model classifier with open-set rejection."""
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.core.config import settings


class CameraClassifier:
    def __init__(self, model_dir: Path | None = None):
        self.model_dir = model_dir or settings.model_dir / "camera"
        self.registry_path = settings.model_dir / "registry.json"
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names: list[str] = []
        self.trained = False
        self.n_training_samples = 0
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            registry = json.loads(self.registry_path.read_text())
            info = registry.get("camera_classifier", {})
            self.trained = info.get("trained", False)
            self.n_training_samples = int(info.get("n_training_samples", 0))

        model_path = self.model_dir / "model.joblib"
        scaler_path = self.model_dir / "scaler.joblib"
        le_path = self.model_dir / "label_encoder.joblib"
        schema_path = self.model_dir / "feature_schema.json"

        if model_path.exists() and scaler_path.exists() and le_path.exists():
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(le_path)
            self.trained = True
            if schema_path.exists():
                schema = json.loads(schema_path.read_text())
                self.feature_names = schema.get("feature_names", [])

    def predict(self, feature_vector: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
        if self.n_training_samples < settings.min_camera_training_samples:
            return {
                "status": "INSUFFICIENT_TRAINING_DATA",
                "known_camera": False,
                "manufacturer": None,
                "model": "Unknown / Unseen Camera",
                "confidence": 0.0,
                "top_candidates": [],
                "availability": True,
                "message": "Camera ML model has insufficient real training images for reliable attribution.",
            }

        if not self.trained or self.model is None:
            return {
                "status": "MODEL_NOT_TRAINED",
                "known_camera": False,
                "manufacturer": None,
                "model": "Unknown / Unseen Camera",
                "confidence": 0.0,
                "top_candidates": [],
                "availability": False,
            }

        if self.feature_names:
            aligned = np.zeros(len(self.feature_names))
            name_to_idx = {n: i for i, n in enumerate(feature_names)}
            for i, fn in enumerate(self.feature_names):
                if fn in name_to_idx:
                    aligned[i] = feature_vector[name_to_idx[fn]]
            feature_vector = aligned

        X = self.scaler.transform(feature_vector.reshape(1, -1))

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            classes = self.label_encoder.classes_
            top_idx = np.argsort(proba)[::-1][:3]
            top_candidates = [
                {
                    "camera_model": str(classes[i]),
                    "confidence": float(proba[i]),
                    "rank": rank + 1,
                }
                for rank, i in enumerate(top_idx)
            ]
            best_idx = top_idx[0]
            confidence = float(proba[best_idx])
            label = str(classes[best_idx])
        else:
            pred = self.model.predict(X)[0]
            label = str(self.label_encoder.inverse_transform([pred])[0])
            confidence = 0.5
            top_candidates = [{"camera_model": label, "confidence": confidence, "rank": 1}]

        known = confidence >= settings.unknown_camera_threshold
        manufacturer, model_name = self._parse_label(label)

        # Hierarchical classification mode handling
        if settings.camera_classification_mode == "hierarchical" and known:
            # Stage 1: Manufacturer probability aggregation
            mfg_scores: dict[str, float] = {}
            if hasattr(self.model, "predict_proba"):
                classes = self.label_encoder.classes_
                for idx, class_label in enumerate(classes):
                    mfg, _ = self._parse_label(str(class_label))
                    mfg_key = mfg or "Unknown"
                    mfg_scores[mfg_key] = mfg_scores.get(mfg_key, 0.0) + float(proba[idx])
                if mfg_scores:
                    best_mfg = max(mfg_scores, key=mfg_scores.get)
                    if manufacturer and best_mfg != manufacturer and mfg_scores[best_mfg] > 0.5:
                        manufacturer = best_mfg

        return {
            "status": "success" if known else "UNKNOWN_CAMERA",
            "known_camera": known,
            "manufacturer": manufacturer if known else None,
            "model": model_name if known else "Unknown / Unseen Camera",
            "full_label": label,
            "confidence": confidence,
            "classification_mode": settings.camera_classification_mode,
            "uncertainty": float(1.0 - confidence),
            "top_candidates": top_candidates,
            "availability": True,
        }

    @staticmethod
    def _parse_label(label: str) -> tuple[str | None, str | None]:
        parts = label.split("_", 1) if "_" in label else label.split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, label

