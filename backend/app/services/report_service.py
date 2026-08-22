"""Report generation service."""
import json
from pathlib import Path

from app.core.config import settings
from app.forensic.reporting.forensic_report import generate_forensic_report, human_readable_summary


class ReportService:
    def generate(self, analysis_id: str, result: dict) -> dict:
        report = generate_forensic_report(
            analysis_id=analysis_id,
            evidence=result.get("evidence", {}),
            metadata=result.get("metadata", {}),
            camera=result.get("camera", {}),
            manipulation=result.get("manipulation", {}),
            consistency=result.get("consistency", {}),
            robustness=result.get("robustness"),
            artifacts=result.get("artifacts", {}),
        )

        report_dir = settings.artifact_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / f"{analysis_id}.json"
        txt_path = report_dir / f"{analysis_id}.txt"

        json_path.write_text(json.dumps(report, indent=2, default=str))
        txt_path.write_text(human_readable_summary(report))

        report["artifact_paths"] = {
            "json": str(json_path),
            "text": str(txt_path),
        }
        return report
