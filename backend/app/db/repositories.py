"""Database repository layer."""
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.models import (
    Analysis,
    CameraPrediction,
    Case,
    Evidence,
    ForensicArtifact,
    ManipulationResult,
    RobustnessResult,
)


class AnalysisRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_analysis(self, analysis_id: str, case_db_id: int | None = None) -> Analysis:
        a = Analysis(analysis_id=analysis_id, case_id=case_db_id, status="QUEUED")
        self.db.add(a)
        self.db.commit()
        self.db.refresh(a)
        return a

    def get_by_analysis_id(self, analysis_id: str) -> Analysis | None:
        return self.db.query(Analysis).filter(Analysis.analysis_id == analysis_id).first()

    def update_status(self, analysis_id: str, status: str, progress: float = 0.0, error: str | None = None) -> None:
        a = self.get_by_analysis_id(analysis_id)
        if a:
            a.status = status
            a.progress = progress
            if error:
                a.error = error
            if status == "COMPLETED":
                a.completed_at = datetime.now(timezone.utc)
            self.db.commit()

    def save_result(self, analysis_id: str, result: dict) -> None:
        a = self.get_by_analysis_id(analysis_id)
        if a:
            a.result_json = result
            a.status = "COMPLETED"
            a.progress = 100.0
            a.completed_at = datetime.now(timezone.utc)
            self.db.commit()

    def save_evidence(self, analysis_db_id: int, data: dict) -> Evidence:
        e = Evidence(analysis_id=analysis_db_id, **data)
        self.db.add(e)
        self.db.commit()
        self.db.refresh(e)
        return e

    def save_camera_prediction(self, analysis_db_id: int, data: dict) -> CameraPrediction:
        cp = CameraPrediction(analysis_id=analysis_db_id, **data)
        self.db.add(cp)
        self.db.commit()
        return cp

    def save_manipulation_result(self, analysis_db_id: int, data: dict) -> ManipulationResult:
        mr = ManipulationResult(analysis_id=analysis_db_id, **data)
        self.db.add(mr)
        self.db.commit()
        return mr

    def save_artifact(self, analysis_db_id: int, artifact_type: str, path: str) -> None:
        self.db.add(ForensicArtifact(analysis_id=analysis_db_id, artifact_type=artifact_type, path=path))
        self.db.commit()

    def save_robustness(self, analysis_db_id: int, results: list[dict]) -> None:
        for r in results:
            self.db.add(RobustnessResult(analysis_id=analysis_db_id, **r))
        self.db.commit()


class CaseRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, case_id: str, title: str, description: str | None = None) -> Case:
        c = Case(case_id=case_id, title=title, description=description)
        self.db.add(c)
        self.db.commit()
        self.db.refresh(c)
        return c

    def get_by_case_id(self, case_id: str) -> Case | None:
        return self.db.query(Case).filter(Case.case_id == case_id).first()

    def list_all(self) -> list[Case]:
        return self.db.query(Case).order_by(Case.created_at.desc()).all()
