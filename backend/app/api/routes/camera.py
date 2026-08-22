"""Camera attribution endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.db.repositories import AnalysisRepository
from app.schemas.camera import CameraCandidate, CameraResponse

router = APIRouter(prefix="/analysis", tags=["camera"])


@router.get("/{analysis_id}/camera", response_model=CameraResponse)
def get_camera_result(analysis_id: str, db: Session = Depends(get_db)):
    repo = AnalysisRepository(db)
    analysis = repo.get_by_analysis_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if analysis.camera_prediction:
        cp = analysis.camera_prediction
        result = cp.result_json or {}
        candidates = [
            CameraCandidate(**c) for c in result.get("top_candidates", [])
        ]
        return CameraResponse(
            status=cp.status,
            manufacturer=cp.manufacturer,
            model=cp.model,
            confidence=cp.confidence,
            known_camera=cp.known_camera,
            top_candidates=candidates,
            evidence=result.get("evidence", {}),
        )

    if analysis.result_json and "camera" in analysis.result_json:
        cam = analysis.result_json["camera"]
        return CameraResponse(
            status=cam.get("status", ""),
            manufacturer=cam.get("manufacturer"),
            model=cam.get("model"),
            confidence=cam.get("confidence", 0),
            known_camera=cam.get("known_camera", False),
            top_candidates=[CameraCandidate(**c) for c in cam.get("top_candidates", [])],
            evidence=cam.get("evidence", {}),
        )

    raise HTTPException(status_code=404, detail="Camera analysis not yet available")
