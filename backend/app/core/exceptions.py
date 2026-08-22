"""Custom application exceptions."""
from typing import Any


class CameraTraceError(Exception):
    """Base exception."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(CameraTraceError):
    pass


class ImageValidationError(ValidationError):
    pass


class FileTooLargeError(ValidationError):
    pass


class AnalysisNotFoundError(CameraTraceError):
    pass


class CaseNotFoundError(CameraTraceError):
    pass


class ProcessingError(CameraTraceError):
    pass


class ModelUnavailableError(CameraTraceError):
    pass
