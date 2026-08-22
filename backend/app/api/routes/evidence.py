"""Evidence endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.db.repositories import AnalysisRepository
from app.schemas.evidence import EvidenceResponse

router = APIRouter(prefix="/analysis", tags=["evidence"])


@router.get("/{analysis_id}/evidence", response_model=EvidenceResponse)
def get_evidence(analysis_id: str, db: Session = Depends(get_db)):
    repo = AnalysisRepository(db)
    analysis = repo.get_by_analysis_id(analysis_id)
    if not analysis or not analysis.evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    e = analysis.evidence
    return EvidenceResponse(
        analysis_id=analysis_id,
        filename=e.filename,
        sha256=e.sha256,
        perceptual_hash=e.perceptual_hash,
        width=e.width,
        height=e.height,
        channels=e.channels,
        format=e.format,
        file_size=e.file_size,
        upload_timestamp=e.upload_timestamp.isoformat() if e.upload_timestamp else None,
        original_path=e.original_path,
        metadata=e.metadata_json or {},
    )
