"""Main analysis pipeline orchestration."""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger, log_stage
from app.core.utils import sanitize_for_json
from app.db.repositories import AnalysisRepository
from app.forensic.fusion.consistency_engine import ConsistencyEngine
from app.forensic.fusion.evidence_fusion import fuse_module_evidence
from app.services.camera_service import CameraService
from app.services.manipulation_service import ManipulationService
from app.services.report_service import ReportService
from app.services.robustness_service import RobustnessService

logger = get_logger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)


class AnalysisService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = AnalysisRepository(db)
        self.camera_service = CameraService()
        self.manipulation_service = ManipulationService()
        self.robustness_service = RobustnessService()
        self.report_service = ReportService()
        self.consistency_engine = ConsistencyEngine()

    def run_analysis(self, analysis_id: str) -> dict:
        analysis = self.repo.get_by_analysis_id(analysis_id)
        if not analysis or not analysis.evidence:
            raise ValueError(f"Analysis {analysis_id} not found or no evidence")

        image_path = Path(analysis.evidence.original_path)
        artifact_dir = settings.artifact_dir / analysis_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        stages = [
            ("METADATA_ANALYSIS", 10),
            ("CAMERA_FORENSICS", 40),
            ("MANIPULATION_ANALYSIS", 70),
            ("CONSISTENCY_ANALYSIS", 85),
            ("ROBUSTNESS_ANALYSIS", 92),
            ("REPORT_GENERATION", 98),
        ]

        result: dict = {}
        metadata = analysis.evidence.metadata_json or {}

        try:
            for stage, progress in stages:
                self.repo.update_status(analysis_id, stage, progress)
                t0 = time.perf_counter()

                if stage == "METADATA_ANALYSIS":
                    result["metadata"] = metadata
                    result["evidence"] = {
                        "filename": analysis.evidence.filename,
                        "sha256": analysis.evidence.sha256,
                        "perceptual_hash": analysis.evidence.perceptual_hash,
                        "dimensions": f"{analysis.evidence.width}x{analysis.evidence.height}",
                    }

                elif stage == "CAMERA_FORENSICS":
                    camera = self.camera_service.analyze(image_path, artifact_dir)
                    result["camera"] = camera
                    self.repo.save_camera_prediction(analysis.id, {
                        "manufacturer": camera.get("manufacturer"),
                        "model": camera.get("model"),
                        "confidence": camera.get("confidence", 0),
                        "known_camera": camera.get("known_camera", False),
                        "status": camera.get("status", ""),
                        "result_json": camera,
                    })

                elif stage == "MANIPULATION_ANALYSIS":
                    manipulation = self.manipulation_service.analyze(image_path, artifact_dir)
                    result["manipulation"] = manipulation
                    self.repo.save_manipulation_result(analysis.id, {
                        "status": manipulation.get("status", ""),
                        "overall_suspiciousness": manipulation.get("overall_suspiciousness", 0),
                        "indicators": manipulation.get("indicators", []),
                        "result_json": manipulation,
                    })

                elif stage == "CONSISTENCY_ANALYSIS":
                    consistency = self.consistency_engine.analyze(
                        metadata,
                        result.get("camera", {}),
                        result.get("manipulation", {}),
                        result.get("camera", {}).get("evidence", {}).get("prnu"),
                    )
                    fused = fuse_module_evidence({
                        "camera": result.get("camera", {}),
                        "manipulation": result.get("manipulation", {}),
                        "metadata": {"status": "success", "availability": bool(metadata)},
                    })
                    result["consistency"] = consistency
                    result["evidence_fusion"] = fused

                elif stage == "ROBUSTNESS_ANALYSIS":
                    result["robustness"] = self.robustness_service.analyze(image_path)

                elif stage == "REPORT_GENERATION":
                    artifacts = self._collect_artifacts(artifact_dir)
                    result["artifacts"] = artifacts
                    report = self.report_service.generate(analysis_id, result)
                    result["report"] = report

                log_stage(logger, analysis_id, stage, "completed", (time.perf_counter() - t0) * 1000)

            clean_result = sanitize_for_json(result)
            self.repo.save_result(analysis_id, clean_result)
            self.repo.update_status(analysis_id, "COMPLETED", 100.0)
            return clean_result

        except Exception as e:
            logger.exception("Analysis failed", extra={"analysis_id": analysis_id})
            self.repo.update_status(analysis_id, "FAILED", error=str(e))
            raise

    def run_analysis_async(self, analysis_id: str) -> None:
        _executor.submit(self._run_safe, analysis_id)

    def _run_safe(self, analysis_id: str) -> None:
        from app.db.database import SessionLocal
        db = SessionLocal()
        try:
            AnalysisService(db).run_analysis(analysis_id)
        except Exception:
            pass
        finally:
            db.close()

    def _collect_artifacts(self, artifact_dir: Path) -> dict:
        artifacts = {}
        for p in artifact_dir.iterdir():
            if p.is_file():
                artifacts[p.stem] = str(p)
        return artifacts

    def get_result(self, analysis_id: str) -> dict | None:
        analysis = self.repo.get_by_analysis_id(analysis_id)
        if not analysis:
            return None
        return {
            "analysis_id": analysis_id,
            "status": analysis.status,
            "progress": analysis.progress,
            "error": analysis.error,
            "result": analysis.result_json,
            "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
            "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
        }
