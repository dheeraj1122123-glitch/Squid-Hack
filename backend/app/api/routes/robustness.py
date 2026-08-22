"""Robustness testing endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.db.repositories import AnalysisRepository
from app.schemas.robustness import RobustnessResponse, RobustnessTransformResult

router = APIRouter(prefix="/analysis", tags=["robustness"])


@router.get("/{analysis_id}/robustness", response_model=RobustnessResponse)
def get_robustness_result(analysis_id: str, db: Session = Depends(get_db)):
    repo = AnalysisRepository(db)
    analysis = repo.get_by_analysis_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if analysis.result_json and "robustness" in analysis.result_json:
        rob = analysis.result_json["robustness"]
        return RobustnessResponse(
            enabled=rob.get("enabled", False),
            results=[RobustnessTransformResult(**r) for r in rob.get("results", [])],
            robustness_summary=rob.get("robustness_summary", {}),
        )

    raise HTTPException(status_code=404, detail="Robustness analysis not yet available")
