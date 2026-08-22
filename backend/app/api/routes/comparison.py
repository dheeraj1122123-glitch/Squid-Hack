"""Original-versus-edited image comparison endpoint."""
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.exceptions import CameraTraceError, ImageValidationError
from app.core.security import validate_extension, validate_file_size, validate_mime
from app.services.image_comparison_service import ImageComparisonService

router = APIRouter(prefix="/compare", tags=["comparison"])


@router.post("")
async def compare_images(
    original: UploadFile = File(...),
    edited: UploadFile = File(...),
):
    try:
        for upload in (original, edited):
            validate_extension(upload.filename or "image.jpg")
            validate_mime(upload.content_type)
        original_bytes, edited_bytes = await original.read(), await edited.read()
        validate_file_size(len(original_bytes))
        validate_file_size(len(edited_bytes))
        return ImageComparisonService().compare(original_bytes, edited_bytes)
    except ImageValidationError as error:
        raise HTTPException(status_code=400, detail=error.message) from error
    except CameraTraceError as error:
        raise HTTPException(status_code=422, detail=error.message) from error
