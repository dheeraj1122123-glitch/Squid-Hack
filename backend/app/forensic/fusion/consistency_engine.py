"""Forensic consistency engine."""
from typing import Any

from app.forensic.metadata.consistency import check_metadata_signal_mismatch


class ConsistencyEngine:
    def analyze(
        self,
        metadata: dict,
        camera: dict,
        manipulation: dict,
        prnu: dict | None = None,
        ai_detection: dict | None = None,
    ) -> dict[str, Any]:
        flags: list[str] = []

        mismatch = check_metadata_signal_mismatch(
            metadata,
            camera.get("manufacturer"),
            camera.get("model"),
        )
        if mismatch["metadata_signal_mismatch"]:
            flags.append("METADATA_SIGNAL_MISMATCH")

        if not camera.get("known_camera", False) and camera.get("status") == "UNKNOWN_CAMERA":
            flags.append("UNKNOWN_CAMERA")

        if camera.get("status") == "MODEL_NOT_TRAINED":
            flags.append("MODEL_UNAVAILABLE")

        manip_status = manipulation.get("status", "")
        if manipulation.get("local_noise_anomaly_score", 0) > 0.5:
            flags.append("LOCAL_NOISE_INCONSISTENCY")
        if manipulation.get("compression_inconsistency_score", 0) > 0.4:
            flags.append("COMPRESSION_INCONSISTENCY")
        if manipulation.get("copy_move_detected"):
            flags.append("POSSIBLE_COPY_MOVE")
        if manipulation.get("tampered_probability", 0) > 0.5:
            flags.append("POSSIBLE_SPLICE")
        if manipulation.get("possible_recompression"):
            flags.append("POSSIBLE_RECOMPRESSION")

        if prnu and not prnu.get("prnu_reference_available"):
            pass  # Not a flag — expected when no reference

        if ai_detection and ai_detection.get("status") == "model_unavailable":
            flags.append("MODEL_UNAVAILABLE")

        overall = self._interpret(flags, camera, manipulation)

        return {
            "consistency_flags": flags,
            "metadata_signal_mismatch": mismatch["metadata_signal_mismatch"],
            "mismatch_reasons": mismatch.get("mismatch_reasons", []),
            "overall_assessment": overall,
            "camera_result": camera,
            "manipulation_result": manipulation,
            "uncertainty": self._compute_uncertainty(camera, manipulation),
        }

    def _interpret(self, flags: list[str], camera: dict, manipulation: dict) -> str:
        if "MODEL_UNAVAILABLE" in flags and camera.get("status") == "MODEL_NOT_TRAINED":
            return "inconclusive — camera model not trained"
        manip_flags = {"POSSIBLE_COPY_MOVE", "POSSIBLE_SPLICE", "LOCAL_NOISE_INCONSISTENCY", "COMPRESSION_INCONSISTENCY"}
        if flags & manip_flags if isinstance(flags, set) else any(f in manip_flags for f in flags):
            if len([f for f in flags if f in manip_flags]) >= 2:
                return "potentially manipulated"
            return "suspicious — further investigation recommended"
        if "METADATA_SIGNAL_MISMATCH" in flags:
            return "suspicious — metadata and signal analysis disagree"
        if not flags:
            return "consistent — no major inconsistencies detected"
        return "inconclusive"

    def _compute_uncertainty(self, camera: dict, manipulation: dict) -> float:
        cam_unc = camera.get("uncertainty", 0.5)
        manip_unc = 1.0 - manipulation.get("overall_suspiciousness", 0.0)
        if camera.get("status") == "MODEL_NOT_TRAINED":
            return 0.9
        return float((cam_unc + manip_unc) / 2)
