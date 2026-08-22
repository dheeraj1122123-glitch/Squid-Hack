"""EXIF and metadata extraction."""
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def _decode_exif_value(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    if isinstance(value, tuple):
        return tuple(_decode_exif_value(v) for v in value)
    return value


def extract_exif(path: Path) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return fields
            for tag_id, val in exif.items():
                tag = TAGS.get(tag_id, str(tag_id))
                fields[tag] = _decode_exif_value(val)
            gps_ifd = exif.get_ifd(0x8825)
            if gps_ifd:
                gps = {}
                for k, v in gps_ifd.items():
                    gps[GPSTAGS.get(k, str(k))] = _decode_exif_value(v)
                fields["GPSInfo"] = gps
    except Exception:
        pass
    return fields


def extract_metadata(path: Path) -> dict[str, Any]:
    exif = extract_exif(path)
    make = exif.get("Make", "")
    model = exif.get("Model", "")
    software = exif.get("Software", "")
    flags: list[str] = []

    if not exif:
        flags.append("NO_EXIF")
    if make and model and make.lower() in model.lower():
        pass
    elif make and model:
        pass
    if software and any(s in str(software).lower() for s in ("photoshop", "gimp", "lightroom", "snapseed")):
        flags.append("EDITING_SOFTWARE_DETECTED")

    gps_present = "GPSInfo" in exif

    return {
        "metadata_available": bool(exif),
        "metadata_fields": exif,
        "metadata_camera_make": str(make) if make else None,
        "metadata_camera_model": str(model) if model else None,
        "metadata_software": str(software) if software else None,
        "metadata_iso": exif.get("ISOSpeedRatings") or exif.get("PhotographicSensitivity"),
        "metadata_exposure": exif.get("ExposureTime"),
        "metadata_focal_length": exif.get("FocalLength"),
        "metadata_orientation": exif.get("Orientation"),
        "metadata_datetime": exif.get("DateTimeOriginal") or exif.get("DateTime"),
        "metadata_gps_present": gps_present,
        "metadata_suspicious_flags": flags,
    }
