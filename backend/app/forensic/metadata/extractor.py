"""Extract embedded image metadata from EXIF, XMP and format-native chunks."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS


CAMERA_MAKE_KEYS = {"make", "cameramake"}
CAMERA_MODEL_KEYS = {"model", "cameramodel"}
SOFTWARE_KEYS = {"software", "creatortool", "processingsoftware"}


def _decode(value: Any) -> Any:
    """Return JSON-safe metadata values without losing readable text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _decode(item) for key, item in value.items()}
    return str(value)


def extract_exif(path: Path) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            if not exif:
                return fields
            for tag_id, value in exif.items():
                fields[TAGS.get(tag_id, str(tag_id))] = _decode(value)
            gps_ifd = exif.get_ifd(0x8825)
            if gps_ifd:
                fields["GPSInfo"] = {
                    GPSTAGS.get(key, str(key)): _decode(value)
                    for key, value in gps_ifd.items()
                }
    except Exception:
        # Metadata parsing must never make an otherwise valid upload fail.
        pass
    return fields


def _xmp_packets(path: Path, image_info: dict[str, Any]) -> list[str]:
    packets: list[str] = []
    for key in ("xmp", "XML:com.adobe.xmp", "XML:com.adobe.xmpmeta"):
        value = image_info.get(key)
        if value:
            packets.append(str(_decode(value)))
    try:
        # Pillow does not expose XMP consistently for JPEG/WebP, so scan a
        # bounded prefix for a complete standard XMP packet.
        text = path.read_bytes()[:8_000_000].decode("utf-8", errors="ignore")
        packets.extend(match.group(0) for match in re.finditer(
            r"<x:xmpmeta[\\s\\S]*?</x:xmpmeta>", text, re.IGNORECASE
        ))
    except OSError:
        pass
    return list(dict.fromkeys(packet for packet in packets if packet))


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1].split(":")[-1]


def _extract_xmp(path: Path, image_info: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for packet in _xmp_packets(path, image_info):
        try:
            root = ET.fromstring(packet)
        except ET.ParseError:
            continue
        for element in root.iter():
            for name, value in element.attrib.items():
                key = _local_name(name)
                if value and key not in fields:
                    fields[key] = _decode(value)
            text = (element.text or "").strip()
            key = _local_name(element.tag)
            if text and key not in {"xmpmeta", "RDF", "Description"} and key not in fields:
                fields[key] = text
    return fields


def _extract_native_fields(image_info: dict[str, Any]) -> dict[str, Any]:
    # These are structural decoder fields, present on ordinary re-saved files;
    # they are not creator-provided metadata evidence.
    ignored = {
        "exif", "xmp", "icc_profile", "XML:com.adobe.xmp", "XML:com.adobe.xmpmeta",
        "jfif", "jfif_version", "jfif_unit", "jfif_density", "progressive", "progression",
        "loop", "background", "duration", "timestamp",
    }
    fields: dict[str, Any] = {}
    for key, value in image_info.items():
        if key in ignored or value is None:
            continue
        decoded = _decode(value)
        if isinstance(decoded, str) and decoded.strip() and len(decoded) <= 4_096:
            fields[str(key)] = decoded
    return fields


def _find_value(key_set: set[str], *groups: dict[str, Any]) -> Any:
    for group in groups:
        for key, value in group.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in key_set and value not in (None, ""):
                return value
    return None


def extract_metadata(path: Path) -> dict[str, Any]:
    """Return all supported embedded metadata and normalized camera evidence."""
    exif = extract_exif(path)
    image_info: dict[str, Any] = {}
    try:
        with Image.open(path) as image:
            image_info = dict(image.info)
    except Exception:
        pass

    xmp = _extract_xmp(path, image_info)
    native = _extract_native_fields(image_info)
    make = _find_value(CAMERA_MAKE_KEYS, exif, xmp, native)
    model = _find_value(CAMERA_MODEL_KEYS, exif, xmp, native)
    software = _find_value(SOFTWARE_KEYS, exif, xmp, native)
    sources = [source for source, fields in (("EXIF", exif), ("XMP", xmp), ("CONTAINER", native)) if fields]
    flags: list[str] = []
    if not sources:
        flags.append("NO_EMBEDDED_METADATA")
    if software and any(name in str(software).lower() for name in ("photoshop", "gimp", "lightroom", "snapseed")):
        flags.append("EDITING_SOFTWARE_DETECTED")

    fields = {
        **{f"EXIF:{key}": value for key, value in exif.items()},
        **{f"XMP:{key}": value for key, value in xmp.items()},
        **{f"CONTAINER:{key}": value for key, value in native.items()},
    }
    return {
        "metadata_available": bool(sources),
        "metadata_sources": sources,
        "metadata_fields": fields,
        "metadata_camera_make": str(make) if make else None,
        "metadata_camera_model": str(model) if model else None,
        "metadata_software": str(software) if software else None,
        "metadata_iso": _find_value({"isospeedratings", "photographicsensitivity", "iso"}, exif, xmp, native),
        "metadata_exposure": _find_value({"exposuretime", "exposure"}, exif, xmp, native),
        "metadata_focal_length": _find_value({"focallength"}, exif, xmp, native),
        "metadata_orientation": _find_value({"orientation"}, exif, xmp, native),
        "metadata_datetime": _find_value({"datetimeoriginal", "datetime", "datecreated", "createdate"}, exif, xmp, native),
        "metadata_gps_present": "GPSInfo" in exif or any("gps" in key.lower() for key in xmp),
        "metadata_suspicious_flags": flags,
    }
