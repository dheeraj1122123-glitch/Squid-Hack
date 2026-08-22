"""Pydantic schemas for analysis."""
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AnalysisStatus(str, Enum):
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    METADATA_ANALYSIS = "METADATA_ANALYSIS"
    CAMERA_FORENSICS = "CAMERA_FORENSICS"
    MANIPULATION_ANALYSIS = "MANIPULATION_ANALYSIS"
    CONSISTENCY_ANALYSIS = "CONSISTENCY_ANALYSIS"
    ROBUSTNESS_ANALYSIS = "ROBUSTNESS_ANALYSIS"
    REPORT_GENERATION = "REPORT_GENERATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class UploadResponse(BaseModel):
    analysis_id: str
    filename: str
    sha256: str
    width: int
    height: int
    format: str
    file_size: int
    status: AnalysisStatus = AnalysisStatus.QUEUED


class AnalysisStatusResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    progress: float = 0.0
    message: str | None = None
    error: str | None = None


class AnalysisResultResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    created_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
