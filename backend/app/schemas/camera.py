"""Camera attribution schemas."""
from typing import Any

from pydantic import BaseModel


class CameraCandidate(BaseModel):
    camera_model: str
    confidence: float
    rank: int


class CameraResponse(BaseModel):
    status: str
    manufacturer: str | None = None
    model: str | None = None
    confidence: float = 0.0
    known_camera: bool = False
    top_candidates: list[CameraCandidate] = []
    evidence: dict[str, Any] = {}
