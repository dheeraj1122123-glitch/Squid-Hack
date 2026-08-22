"""Evidence intake and validation service."""
import shutil
from datetime import datetime, timezone
from pathlib import Path

import imagehash
from PIL import Image

from app.core.config import settings
from app.core.exceptions import ImageValidationError
from app.core.security import (
    generate_analysis_id,
    safe_filename,
    sha256_bytes,
    validate_extension,
    validate_file_size,
    validate_mime,
)
from app.forensic.metadata.extractor import extract_metadata
from app.forensic.preprocessing.image_loader import get_image_info


class EvidenceService:
    def intake(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str | None = None,
        analysis_id: str | None = None,
    ) -> dict:
        validate_file_size(len(file_bytes))
        validate_extension(filename)
        validate_mime(content_type)

        aid = analysis_id or generate_analysis_id()
        ext = Path(filename).suffix.lstrip(".").lower()
        upload_dir = settings.upload_dir / aid
        upload_dir.mkdir(parents=True, exist_ok=True)
        original_path = upload_dir / f"original.{ext}"

        with open(original_path, "wb") as f:
            f.write(file_bytes)

        try:
            with Image.open(original_path) as im:
                im.verify()
            with Image.open(original_path) as im:
                im.load()
                phash = str(imagehash.phash(im))
        except Exception as e:
            original_path.unlink(missing_ok=True)
            raise ImageValidationError(f"Corrupted or invalid image: {e}") from e

        info = get_image_info(original_path)
        file_hash = sha256_bytes(file_bytes)
        metadata = extract_metadata(original_path)

        return {
            "analysis_id": aid,
            "filename": safe_filename(filename),
            "sha256": file_hash,
            "perceptual_hash": phash,
            "width": info["width"],
            "height": info["height"],
            "channels": info["channels"],
            "format": info["format"],
            "file_size": len(file_bytes),
            "upload_timestamp": datetime.now(timezone.utc).isoformat(),
            "original_path": str(original_path),
            "metadata": metadata,
        }
