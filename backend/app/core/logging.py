"""Structured logging configuration."""
import logging
import sys
from typing import Any

from app.core.config import settings


class AnalysisContextFilter(logging.Filter):
    """Inject analysis_id into log records when available."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "analysis_id"):
            record.analysis_id = "-"
        return True


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | analysis=%(analysis_id)s | %(name)s | %(message)s"
        )
    )
    handler.addFilter(AnalysisContextFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_stage(
    logger: logging.Logger,
    analysis_id: str,
    stage: str,
    status: str,
    duration_ms: float | None = None,
    **extra: Any,
) -> None:
    msg = f"stage={stage} status={status}"
    if duration_ms is not None:
        msg += f" duration_ms={duration_ms:.1f}"
    for k, v in extra.items():
        msg += f" {k}={v}"
    logger.info(msg, extra={"analysis_id": analysis_id})
