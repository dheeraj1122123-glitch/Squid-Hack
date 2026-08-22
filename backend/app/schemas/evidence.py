"""Evidence schemas."""
from typing import Any

from pydantic import BaseModel


class EvidenceResponse(BaseModel):
    analysis_id: str
    filename: str
    sha256: str
    perceptual_hash: str | None = None
    width: int
    height: int
    channels: int
    format: str
    file_size: int
    upload_timestamp: str | None = None
    original_path: str
    metadata: dict[str, Any] = {}
