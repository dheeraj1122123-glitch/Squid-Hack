"""Report schemas."""
from typing import Any

from pydantic import BaseModel


class ReportResponse(BaseModel):
    analysis_id: str
    report: dict[str, Any]
    human_readable: str | None = None
