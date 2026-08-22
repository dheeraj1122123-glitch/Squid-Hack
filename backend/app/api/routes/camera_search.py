"""Camera search endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.db.repositories import AnalysisRepository
from app.forensic.camera.similarity import SimilaritySearch

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("/search")
def search_cameras(q: str = "", top_k: int = 5, db: Session = Depends(get_db)):
    search = SimilaritySearch()
    if not search.labels:
        return {"results": [], "message": "No camera embeddings indexed. Train models first."}
    return {"query": q, "results": search.labels[:top_k]}


@router.get("/{camera_model}")
def get_camera_info(camera_model: str):
    from app.forensic.camera.fingerprint import FingerprintStore
    from app.core.config import settings

    store = FingerprintStore(settings.model_dir / "camera" / "fingerprints")
    fp = store.get_fingerprint(camera_model)
    return {
        "camera_model": camera_model,
        "fingerprint_available": fp is not None,
        "registered_cameras": store.list_cameras(),
    }
