"""Health check endpoint."""
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "CameraTrace",
        "version": "1.0.0",
    }
