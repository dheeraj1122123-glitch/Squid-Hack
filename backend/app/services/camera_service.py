"""Camera forensics orchestration service."""
from pathlib import Path

import numpy as np

from app.forensic.camera.cfa import extract_cfa_features
from app.forensic.camera.classifier import CameraClassifier
from app.forensic.camera.features import build_feature_vector
from app.forensic.camera.frequency import frequency_analysis
from app.forensic.camera.jpeg import analyze_compression
from app.forensic.camera.noise import analyze_noise
from app.forensic.camera.prnu import analyze_prnu
from app.forensic.camera.similarity import SimilaritySearch
from app.forensic.metadata.extractor import extract_metadata
from app.forensic.preprocessing.image_loader import load_image_rgb


class CameraService:
    def __init__(self):
        self.classifier = CameraClassifier()
        self.similarity = SimilaritySearch()

    def analyze(self, image_path: Path, artifact_dir: Path | None = None) -> dict:
        img = load_image_rgb(image_path)

        noise = analyze_noise(img, artifact_dir)
        cfa = extract_cfa_features(img)
        freq = frequency_analysis(img, artifact_dir)
        compression = analyze_compression(img, image_path)
        prnu = analyze_prnu(img, reference_fingerprint=None)

        residual = noise.pop("residual", None)
        feature_vector, feature_names = build_feature_vector(noise, cfa, freq, compression, prnu)
        prediction = self.classifier.predict(feature_vector, feature_names)
        metadata = extract_metadata(image_path)

        # Embedded camera fields are direct evidence, whether they originate in
        # EXIF, XMP, PNG text, or another supported image container.
        make = metadata.get("metadata_camera_make")
        model = metadata.get("metadata_camera_model")
        if make or model:
            label = " ".join(part for part in (make, model) if part)
            prediction.update({
                "status": "METADATA_CAMERA_IDENTIFIED",
                "known_camera": True,
                "manufacturer": make,
                "model": model or label,
                "full_label": label,
                "confidence": 0.95,
                "uncertainty": 0.05,
                "prediction_source": "EMBEDDED_METADATA",
            })
        else:
            prediction.setdefault("prediction_source", "FORENSIC_ML")

        similar = self.similarity.search(feature_vector, top_k=5)

        return {
            **prediction,
            "evidence": {
                "noise": {k: v for k, v in noise.items() if k != "residual"},
                "prnu": prnu,
                "cfa": cfa,
                "frequency": freq,
                "compression": compression,
                "metadata": metadata,
            },
            "similar_cameras": similar,
            "feature_vector": feature_vector.tolist(),
            "feature_names": feature_names,
        }
