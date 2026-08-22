"""Forensic report generation."""
from datetime import datetime, timezone
from typing import Any


def generate_forensic_report(
    analysis_id: str,
    evidence: dict,
    metadata: dict,
    camera: dict,
    manipulation: dict,
    consistency: dict,
    robustness: dict | None = None,
    artifacts: dict | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []

    if camera.get("status") == "MODEL_NOT_TRAINED":
        warnings.append("Camera classifier not trained — attribution unavailable")
    if manipulation.get("learned_detector_status") == "MODEL_NOT_TRAINED":
        warnings.append("Learned manipulation detector not trained")
    warnings.extend(manipulation.get("limitations", []))

    return {
        "analysis_id": analysis_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence": evidence,
        "metadata": {
            "section": "FACTUAL OBSERVATIONS",
            **metadata,
        },
        "camera_attribution": {
            "section": "MODEL PREDICTIONS",
            **camera,
        },
        "manipulation_analysis": {
            "section": "FORENSIC INTERPRETATION",
            **manipulation,
        },
        "consistency": consistency,
        "robustness": robustness or {"enabled": False},
        "artifacts": artifacts or {},
        "warnings": warnings,
        "limitations": [
            "Results are forensic indicators, not definitive proof",
            "Metadata is not treated as ground truth",
            "PRNU requires reference fingerprints from the same device",
        ],
    }


def human_readable_summary(report: dict) -> str:
    lines = [
        f"CameraTrace Forensic Report — Analysis {report['analysis_id']}",
        f"Generated: {report['generated_at']}",
        "",
        "=== METADATA (Factual Observations) ===",
    ]
    meta = report.get("metadata", {})
    lines.append(f"  Camera Make: {meta.get('metadata_camera_make', 'N/A')}")
    lines.append(f"  Camera Model: {meta.get('metadata_camera_model', 'N/A')}")
    lines.append(f"  EXIF Available: {meta.get('metadata_available', False)}")

    lines.extend(["", "=== CAMERA ATTRIBUTION (Model Predictions) ==="])
    cam = report.get("camera_attribution", {})
    lines.append(f"  Status: {cam.get('status', 'N/A')}")
    lines.append(f"  Manufacturer: {cam.get('manufacturer', 'N/A')}")
    lines.append(f"  Model: {cam.get('model', 'N/A')}")
    lines.append(f"  Confidence: {cam.get('confidence', 0):.3f}")

    lines.extend(["", "=== MANIPULATION ANALYSIS ==="])
    manip = report.get("manipulation_analysis", {})
    lines.append(f"  Status: {manip.get('status', 'N/A')}")
    lines.append(f"  Suspiciousness: {manip.get('overall_suspiciousness', 0):.3f}")

    lines.extend(["", "=== CONSISTENCY ==="])
    cons = report.get("consistency", {})
    lines.append(f"  Assessment: {cons.get('overall_assessment', 'N/A')}")
    lines.append(f"  Flags: {', '.join(cons.get('consistency_flags', [])) or 'None'}")

    if report.get("warnings"):
        lines.extend(["", "=== WARNINGS ==="])
        for w in report["warnings"]:
            lines.append(f"  - {w}")

    return "\n".join(lines)
