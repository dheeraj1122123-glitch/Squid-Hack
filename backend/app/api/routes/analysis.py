"""Analysis upload and pipeline endpoints."""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.core.exceptions import CameraTraceError, ImageValidationError
from app.db.repositories import AnalysisRepository
from app.schemas.analysis import AnalysisResultResponse, AnalysisStatusResponse, UploadResponse
from app.services.analysis_service import AnalysisService
from app.services.evidence_service import EvidenceService

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        content = await file.read()
        evidence_svc = EvidenceService()
        intake = evidence_svc.intake(content, file.filename or "upload.jpg", file.content_type)

        repo = AnalysisRepository(db)
        analysis = repo.create_analysis(intake["analysis_id"])
        repo.save_evidence(analysis.id, {
            "filename": intake["filename"],
            "sha256": intake["sha256"],
            "perceptual_hash": intake["perceptual_hash"],
            "width": intake["width"],
            "height": intake["height"],
            "channels": intake["channels"],
            "format": intake["format"],
            "file_size": intake["file_size"],
            "original_path": intake["original_path"],
            "metadata_json": intake["metadata"],
        })

        return UploadResponse(
            analysis_id=intake["analysis_id"],
            filename=intake["filename"],
            sha256=intake["sha256"],
            width=intake["width"],
            height=intake["height"],
            format=intake["format"],
            file_size=intake["file_size"],
        )
    except ImageValidationError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except CameraTraceError as e:
        raise HTTPException(status_code=422, detail=e.message)


@router.post("/{analysis_id}/run")
def run_analysis(analysis_id: str, sync: bool = False, db: Session = Depends(get_db)):
    repo = AnalysisRepository(db)
    analysis = repo.get_by_analysis_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    svc = AnalysisService(db)
    if sync:
        res = svc.run_analysis(analysis_id)
        return {"analysis_id": analysis_id, "status": "COMPLETED", "result": res}
    else:
        svc.run_analysis_async(analysis_id)
        return {"analysis_id": analysis_id, "status": "QUEUED", "message": "Analysis started"}


@router.get("/{analysis_id}", response_model=AnalysisResultResponse)
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    svc = AnalysisService(db)
    result = svc.get_result(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return AnalysisResultResponse(**result)


@router.get("/{analysis_id}/status", response_model=AnalysisStatusResponse)
def get_status(analysis_id: str, db: Session = Depends(get_db)):
    repo = AnalysisRepository(db)
    analysis = repo.get_by_analysis_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return AnalysisStatusResponse(
        analysis_id=analysis_id,
        status=analysis.status,
        progress=analysis.progress,
        error=analysis.error,
    )
