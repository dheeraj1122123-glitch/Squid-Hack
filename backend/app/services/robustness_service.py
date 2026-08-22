"""Robustness testing service."""
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.camera_service import CameraService


class RobustnessService:
    def __init__(self):
        self.camera_service = CameraService()

    def analyze(self, image_path: Path) -> dict:
        if not settings.enable_robustness:
            return {"enabled": False, "results": [], "robustness_summary": {}}

        img = cv2.cvtColor(
            cv2.imread(str(image_path)),
            cv2.COLOR_BGR2RGB,
        )
        baseline = self.camera_service.analyze(image_path)
        baseline_label = baseline.get("full_label") or baseline.get("model", "unknown")
        baseline_conf = baseline.get("confidence", 0)

        transforms = self._generate_transforms(img, image_path)
        results = []

        for name, transformed_path in transforms:
            pred = self.camera_service.analyze(transformed_path)
            label = pred.get("full_label") or pred.get("model", "unknown")
            conf = pred.get("confidence", 0)
            changed = label != baseline_label
            results.append({
                "transformation": name,
                "prediction": label,
                "confidence": conf,
                "prediction_changed": changed,
            })
            transformed_path.unlink(missing_ok=True)

        changed_count = sum(1 for r in results if r["prediction_changed"])
        robustness_score = 1.0 - (changed_count / max(len(results), 1))

        return {
            "enabled": True,
            "results": results,
            "robustness_summary": {
                "baseline_prediction": baseline_label,
                "baseline_confidence": baseline_conf,
                "transformations_tested": len(results),
                "predictions_changed": changed_count,
                "robustness_score": robustness_score,
            },
        }

    def _generate_transforms(self, img: np.ndarray, original_path: Path) -> list[tuple[str, Path]]:
        import uuid
        tmp_dir = settings.artifact_dir / "robustness_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        transforms: list[tuple[str, Path]] = []
        uid = uuid.uuid4().hex[:8]

        for quality in (90, 70, 50):
            p = tmp_dir / f"{uid}_jpeg_q{quality}.jpg"
            Image.fromarray(img).save(p, format="JPEG", quality=quality)
            transforms.append((f"jpeg_quality_{quality}", p))

        for scale, label in [(0.75, "75pct"), (0.5, "50pct")]:
            h, w = img.shape[:2]
            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
            p = tmp_dir / f"{uid}_resize_{label}.png"
            cv2.imwrite(str(p), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            transforms.append((f"resize_{label}", p))

        bright = np.clip(img.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
        p = tmp_dir / f"{uid}_brightness.png"
        cv2.imwrite(str(p), cv2.cvtColor(bright, cv2.COLOR_RGB2BGR))
        transforms.append(("brightness_up_10pct", p))

        return transforms
