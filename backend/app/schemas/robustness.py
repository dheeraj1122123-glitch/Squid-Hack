"""Robustness testing schemas."""
from typing import Any

from pydantic import BaseModel


class RobustnessTransformResult(BaseModel):
    transformation: str
    prediction: str | None = None
    confidence: float = 0.0
    prediction_changed: bool = False


class RobustnessResponse(BaseModel):
    enabled: bool
    results: list[RobustnessTransformResult] = []
    robustness_summary: dict[str, Any] = {}
