"""Manipulation detection orchestration service."""
from pathlib import Path

from app.forensic.manipulation.copy_move import detect_copy_move
from app.forensic.manipulation.ela import analyze_ela
from app.forensic.manipulation.jpeg_analysis import analyze_jpeg_manipulation
from app.forensic.manipulation.localization import fuse_localization_maps, save_heatmap
from app.forensic.manipulation.manipulation_classifier import ManipulationClassifier
from app.forensic.manipulation.noise_inconsistency import local_noise_inconsistency, save_noise_heatmap
from app.forensic.manipulation.splice import detect_splicing
from app.forensic.preprocessing.image_loader import load_image_rgb


class ManipulationService:
    def __init__(self):
        self.classifier = ManipulationClassifier()

    def analyze(self, image_path: Path, artifact_dir: Path | None = None) -> dict:
        img = load_image_rgb(image_path)
        h, w = img.shape[:2]

        ela = analyze_ela(image_path, artifact_dir)
        noise = local_noise_inconsistency(img)
        copy_move = detect_copy_move(img)
        splice = detect_splicing(img, image_path)
        jpeg = analyze_jpeg_manipulation(img, image_path)

        localization = fuse_localization_maps(
            h, w,
            ela_result=ela,
            noise_result=noise,
            copy_move_result=copy_move,
            splice_result=splice,
            jpeg_result=jpeg,
        )

        heatmap_path = None
        if artifact_dir:
            heatmap_path = artifact_dir / "suspiciousness_heatmap.png"
            save_heatmap(localization["heatmap"], heatmap_path)
            if noise.get("anomaly_map") is not None:
                save_noise_heatmap(noise["anomaly_map"], artifact_dir / "noise_heatmap.png")

        indicators: list[str] = []
        types: list[str] = []

        if noise.get("local_noise_anomaly_score", 0) > 0.4:
            indicators.append("local_noise_inconsistency")
        if copy_move.get("copy_move_detected"):
            indicators.append("copy_move")
            types.append("COPY_MOVE")
        if splice.get("tampered_probability", 0) > 0.5:
            indicators.append("splicing")
            types.append("SPLICING")
        if jpeg.get("compression_inconsistency_score", 0) > 0.3:
            indicators.append("compression_inconsistency")
            if not types:
                types.append("RECOMPRESSION_ONLY")

        overall = localization["overall_suspiciousness"]
        if overall > 0.5 and not types:
            types.append("UNKNOWN_MANIPULATION")
        if overall < 0.3 and not indicators:
            types.append("AUTHENTIC")

        learned = self.classifier.predict(
            __import__("numpy").array([overall, noise.get("local_noise_anomaly_score", 0), splice.get("tampered_probability", 0)])
        )

        status = "potentially_manipulated" if overall > 0.5 else "suspicious" if overall > 0.3 else "consistent"

        return {
            "status": status,
            "indicators": indicators,
            "types": types,
            "regions": localization["regions"],
            "heatmap": str(heatmap_path) if heatmap_path else None,
            "overall_suspiciousness": overall,
            "local_noise_anomaly_score": noise.get("local_noise_anomaly_score", 0),
            "compression_inconsistency_score": jpeg.get("compression_inconsistency_score", 0),
            "copy_move_detected": copy_move.get("copy_move_detected", False),
            "tampered_probability": splice.get("tampered_probability", 0),
            "possible_recompression": jpeg.get("compression_features", {}).get("possible_recompression", False),
            "learned_detector_status": learned.get("status"),
            "learned_prediction": learned.get("predicted_type"),
            "limitations": [
                "Forensic indicators are not definitive proof of manipulation",
                "ELA alone cannot prove tampering",
            ],
            "evidence": {
                "ela": {k: v for k, v in ela.items() if k != "ela_map"},
                "noise": {k: v for k, v in noise.items() if k != "anomaly_map"},
                "copy_move": {k: v for k, v in copy_move.items() if k != "heatmap"},
                "splice": {k: v for k, v in splice.items() if k != "patch_scores"},
                "jpeg": jpeg,
            },
        }
