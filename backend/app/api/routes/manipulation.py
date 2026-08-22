"""Manipulation analysis endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.db.repositories import AnalysisRepository
from app.schemas.manipulation import ManipulationResponse, SuspiciousRegion

router = APIRouter(prefix="/analysis", tags=["manipulation"])


@router.get("/{analysis_id}/manipulation", response_model=ManipulationResponse)
def get_manipulation_result(analysis_id: str, db: Session = Depends(get_db)):
    repo = AnalysisRepository(db)
    analysis = repo.get_by_analysis_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    manip = None
    if analysis.manipulation_result:
        manip = analysis.manipulation_result.result_json
    elif analysis.result_json:
        manip = analysis.result_json.get("manipulation")

    if not manip:
        raise HTTPException(status_code=404, detail="Manipulation analysis not yet available")

    return ManipulationResponse(
        status=manip.get("status", ""),
        indicators=manip.get("indicators", []),
        types=manip.get("types", []),
        regions=[SuspiciousRegion(**r) for r in manip.get("regions", [])],
        heatmap=manip.get("heatmap"),
        overall_suspiciousness=manip.get("overall_suspiciousness", 0),
        limitations=manip.get("limitations", []),
        evidence=manip.get("evidence", {}),
    )
