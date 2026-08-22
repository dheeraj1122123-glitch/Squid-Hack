"""Case management endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.core.security import generate_analysis_id
from app.db.repositories import AnalysisRepository, CaseRepository
from app.services.evidence_service import EvidenceService

router = APIRouter(prefix="/cases", tags=["cases"])


class CaseCreate(BaseModel):
    title: str
    description: str | None = None


class CaseResponse(BaseModel):
    case_id: str
    title: str
    description: str | None = None


@router.post("", response_model=CaseResponse)
def create_case(body: CaseCreate, db: Session = Depends(get_db)):
    case_id = generate_analysis_id()[:16]
    repo = CaseRepository(db)
    case = repo.create(case_id, body.title, body.description)
    return CaseResponse(case_id=case.case_id, title=case.title, description=case.description)


@router.get("")
def list_cases(db: Session = Depends(get_db)):
    repo = CaseRepository(db)
    cases = repo.list_all()
    return [
        {"case_id": c.case_id, "title": c.title, "description": c.description, "created_at": c.created_at.isoformat()}
        for c in cases
    ]


@router.get("/{case_id}")
def get_case(case_id: str, db: Session = Depends(get_db)):
    repo = CaseRepository(db)
    case = repo.get_by_case_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return {
        "case_id": case.case_id,
        "title": case.title,
        "description": case.description,
        "analyses": [a.analysis_id for a in case.analyses],
    }
