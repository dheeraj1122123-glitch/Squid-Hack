"""Report endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.db.repositories import AnalysisRepository
from app.forensic.reporting.forensic_report import human_readable_summary
from app.schemas.report import ReportResponse

router = APIRouter(prefix="/analysis", tags=["reports"])


@router.get("/{analysis_id}/report", response_model=ReportResponse)
def get_report(analysis_id: str, db: Session = Depends(get_db)):
    repo = AnalysisRepository(db)
    analysis = repo.get_by_analysis_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if analysis.result_json and "report" in analysis.result_json:
        report = analysis.result_json["report"]
        return ReportResponse(
            analysis_id=analysis_id,
            report=report,
            human_readable=human_readable_summary(report),
        )

    raise HTTPException(status_code=404, detail="Report not yet available")


@router.get("/{analysis_id}/artifacts")
def get_artifacts(analysis_id: str, db: Session = Depends(get_db)):
    repo = AnalysisRepository(db)
    analysis = repo.get_by_analysis_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    artifacts = []
    for a in analysis.artifacts:
        artifacts.append({"type": a.artifact_type, "path": a.path})

    if analysis.result_json and "artifacts" in analysis.result_json:
        for name, path in analysis.result_json["artifacts"].items():
            artifacts.append({"type": name, "path": path})

    return {"analysis_id": analysis_id, "artifacts": artifacts}
