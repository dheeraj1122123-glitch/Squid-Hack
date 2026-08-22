"""Manipulation analysis schemas."""
from typing import Any

from pydantic import BaseModel


class SuspiciousRegion(BaseModel):
    x: int
    y: int
    width: int
    height: int
    score: float


class ManipulationResponse(BaseModel):
    status: str
    indicators: list[str] = []
    types: list[str] = []
    regions: list[SuspiciousRegion] = []
    heatmap: str | None = None
    overall_suspiciousness: float = 0.0
    limitations: list[str] = []
    evidence: dict[str, Any] = {}
