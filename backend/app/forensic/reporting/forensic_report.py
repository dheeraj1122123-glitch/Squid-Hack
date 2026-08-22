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
        warnings.append("Camera classifier not trained â€” attribution unavailable")
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
    """Create a concise, traceable report for the web download view."""
    lines = [f"CameraTrace Forensic Report | Analysis {report['analysis_id']}", f"Generated (UTC): {report['generated_at']}", "", "=== EVIDENCE INTAKE ==="]
    evidence = report.get("evidence", {})
    lines.extend([f"  File: {evidence.get('filename', 'N/A')}", f"  SHA-256: {evidence.get('sha256', 'N/A')}", f"  Dimensions: {evidence.get('dimensions', 'N/A')}"])
    meta = report.get("metadata", {})
    lines.extend(["", "=== METADATA OBSERVATIONS ===", f"  EXIF available: {meta.get('metadata_available', False)}", f"  Make / Model: {(meta.get('metadata_camera_make') or 'N/A')} / {(meta.get('metadata_camera_model') or 'N/A')}", f"  Editing software: {meta.get('metadata_software') or 'Not reported'}"])
    cam = report.get("camera_attribution", {})
    lines.extend(["", "=== CAMERA ATTRIBUTION ===", f"  Status: {cam.get('status', 'N/A')}", f"  Result: {cam.get('full_label') or cam.get('model') or 'Unknown'}", f"  Source: {cam.get('prediction_source', 'FORENSIC_ML')}", f"  Confidence: {cam.get('confidence', 0):.1%}", f"  Uncertainty: {cam.get('uncertainty', 1):.1%}"])
    candidates = cam.get("top_candidates", [])
    if candidates:
        lines.append("  Candidates: " + "; ".join(f"{c.get('camera_model')} ({c.get('confidence', 0):.1%})" for c in candidates))
    manip = report.get("manipulation_analysis", {})
    ai = manip.get("ai_generated", {})
    lines.extend(["", "=== MANIPULATION & AI INDICATORS ===", f"  Overall suspiciousness: {manip.get('overall_suspiciousness', 0):.1%}", f"  Findings: {', '.join(manip.get('types', [])) or 'None'}", f"  AI detector: {ai.get('status', 'not available')}"])
    if ai.get("status") == "success": lines.append(f"  AI-generated likelihood: {ai.get('ai_generated_likelihood', 0):.1%}")
    rob = report.get("robustness", {}); summary = rob.get("robustness_summary", {})
    lines.extend(["", "=== TRANSFORMATION STABILITY ===", f"  Enabled: {rob.get('enabled', False)}", f"  Tests: {summary.get('transformations_tested', 0)} | Changed: {summary.get('predictions_changed', 0)} | Stability: {summary.get('robustness_score', 0):.1%}"])
    for item in rob.get("results", []): lines.append(f"  - {item.get('transformation')}: {item.get('prediction', 'Unknown')} ({item.get('confidence', 0):.1%}), changed={item.get('prediction_changed', False)}")
    cons = report.get("consistency", {})
    lines.extend(["", "=== ASSESSMENT ===", f"  {cons.get('overall_assessment', 'inconclusive')}"])
    if report.get("warnings"): lines.extend(["", "=== LIMITATIONS / WARNINGS ==="] + [f"  - {w}" for w in report["warnings"]])
    return "\n".join(lines)