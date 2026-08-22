"""Security utilities for file uploads and path handling."""
import hashlib
import re
import secrets
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import FileTooLargeError, ImageValidationError

ALLOWED_MIME = {
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/bmp",
    "image/webp",
}


def generate_analysis_id() -> str:
    return secrets.token_hex(16)


def safe_filename(name: str) -> str:
    base = Path(name).name
    return re.sub(r"[^\w.\-]", "_", base)[:200]


def validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lstrip(".").lower()
    if ext not in settings.allowed_ext_set:
        raise ImageValidationError(
            f"Extension '.{ext}' not allowed. Allowed: {sorted(settings.allowed_ext_set)}"
        )
    return ext


def validate_file_size(size: int) -> None:
    if size > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"File exceeds maximum size of {settings.max_upload_mb} MB"
        )


def validate_mime(mime: str | None) -> None:
    if mime and mime not in ALLOWED_MIME:
        raise ImageValidationError(f"MIME type '{mime}' not allowed")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_artifact_path(base: Path, *parts: str) -> Path:
    """Prevent path traversal."""
    resolved = base.resolve()
    target = (base / Path(*parts)).resolve()
    if not str(target).startswith(str(resolved)):
        raise ImageValidationError("Invalid artifact path")
    return target
