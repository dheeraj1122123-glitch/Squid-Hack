"""Camera fingerprint storage and management."""
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.forensic.camera.prnu import aggregate_fingerprint, save_fingerprint
from app.forensic.preprocessing.image_loader import load_image_rgb


class FingerprintStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "index.json"
        self._index: dict[str, Any] = self._load_index()

    def _load_index(self) -> dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {"cameras": {}}

    def _save_index(self) -> None:
        self.index_path.write_text(json.dumps(self._index, indent=2))

    def register_camera(self, camera_model: str, image_paths: list[Path]) -> None:
        images = [load_image_rgb(p) for p in image_paths[:20]]
        fp = aggregate_fingerprint(images)
        fp_path = self.base_dir / f"{camera_model.replace('/', '_')}.npy"
        save_fingerprint(fp, fp_path)
        self._index["cameras"][camera_model] = {
            "path": str(fp_path),
            "reference_count": len(image_paths),
        }
        self._save_index()

    def get_fingerprint(self, camera_model: str) -> np.ndarray | None:
        info = self._index.get("cameras", {}).get(camera_model)
        if not info:
            return None
        return np.load(info["path"])

    def list_cameras(self) -> list[str]:
        return list(self._index.get("cameras", {}).keys())
