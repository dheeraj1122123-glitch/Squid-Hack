"""Dataset adapters for forensic datasets."""
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(path: Path) -> str:
    try:
        with Image.open(path) as im:
            return str(imagehash.phash(im))
    except Exception:
        return ""


def parse_dresden_label(path: Path, root: Path) -> dict[str, str | None]:
    """Parse original and Kaggle-exported Dresden model/device folders."""
    parts = path.relative_to(root).parts
    candidates = [p for p in parts[:-1] if re.match(r"^.+_.+?(?:_\d+)?$", p)]
    folder = candidates[-1] if candidates else (parts[-2] if len(parts) > 1 else "")
    match = re.match(r"^(?P<label>.+)_(?P<device>\d+)$", folder)
    label = match.group("label") if match else folder
    device = match.group("device") if match else None
    manufacturer, _, _ = label.partition("_")
    return {"manufacturer": manufacturer or None, "camera_model": label or None,
            "physical_device": device, "dataset": "dresden"}

def parse_casia_label(path: Path, root: Path) -> dict[str, str | None]:
    rel = path.relative_to(root)
    parts = rel.parts
    label = "AUTHENTIC"
    if any(x in str(path).lower() for x in ("tp", "tamper", "fake", "copy")):
        label = "SPLICING"
    elif "cm" in str(path).lower() or "copymove" in str(path).lower():
        label = "COPY_MOVE"
    return {
        "manufacturer": None,
        "camera_model": None,
        "manipulation_type": label,
        "dataset": "casia",
        "scene": parts[0] if parts else None,
    }


def parse_fau_label(path: Path, root: Path) -> dict[str, str | None]:
    rel = path.relative_to(root)
    name = path.stem.lower()
    manip = "AUTHENTIC"
    if "copy" in name or "cm" in name:
        manip = "COPY_MOVE"
    elif "splice" in name or "sp" in name:
        manip = "SPLICING"
    elif "remove" in name or "inpaint" in name:
        manip = "OBJECT_REMOVAL_OR_INPAINTING"
    elif "retouch" in name:
        manip = "RETOUCHING"
    return {
        "manufacturer": None,
        "camera_model": None,
        "manipulation_type": manip,
        "dataset": "fau",
        "scene": rel.parts[0] if rel.parts else None,
    }


def parse_generic_camera(path: Path, root: Path) -> dict[str, str | None]:
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) >= 2:
        return {
            "manufacturer": parts[0],
            "camera_model": f"{parts[0]}_{parts[1]}",
            "physical_device": parts[2] if len(parts) > 2 else None,
            "dataset": "generic",
        }
    return {"manufacturer": None, "camera_model": parts[0] if parts else None, "dataset": "generic"}


DATASET_PARSERS = {
    "dresden": parse_dresden_label,
    "casia": parse_casia_label,
    "fau": parse_fau_label,
    "nist_mfc": parse_generic_camera,
    "vision": parse_generic_camera,
    "generic": parse_generic_camera,
}


def build_manifest(
    root: Path,
    output: Path,
    dataset_type: str = "generic",
    task: str = "camera",
) -> list[dict[str, Any]]:
    parser = DATASET_PARSERS.get(dataset_type, parse_generic_camera)
    images = find_images(root)
    seen_hashes: set[str] = set()
    seen_phashes: dict[str, str] = {}
    manifest: list[dict[str, Any]] = []

    for img_path in tqdm(images, desc=f"Scanning {root.name}"):
        file_hash = sha256_file(img_path)
        if file_hash in seen_hashes:
            continue
        seen_hashes.add(file_hash)

        phash = perceptual_hash(img_path)
        if phash and phash in seen_phashes:
            continue
        if phash:
            seen_phashes[phash] = str(img_path)

        labels = parser(img_path, root)
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                fmt = im.format or "UNKNOWN"
        except Exception:
            continue

        entry = {
            "image_path": str(img_path),
            "sha256": file_hash,
            "perceptual_hash": phash,
            "width": w,
            "height": h,
            "format": fmt,
            **labels,
        }
        if task == "manipulation":
            entry.setdefault("manipulation_type", labels.get("manipulation_type", "AUTHENTIC"))
        manifest.append(entry)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        if manifest:
            writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
            writer.writeheader()
            writer.writerows(manifest)

    meta = {
        "dataset_type": dataset_type,
        "task": task,
        "root": str(root),
        "total_images": len(manifest),
        "duplicates_removed": len(images) - len(manifest),
    }
    output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return manifest
