"""Pixel-level comparison of an original image and an edited derivative."""
from __future__ import annotations

import secrets
from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings
from app.core.exceptions import ImageValidationError


class ImageComparisonService:
    """Align two images, localize visual changes, and save an explainable map."""

    def compare(self, original_bytes: bytes, edited_bytes: bytes) -> dict:
        original = self._decode(original_bytes, "original")
        edited = self._decode(edited_bytes, "edited")
        original_height, original_width = original.shape[:2]
        edited_height, edited_width = edited.shape[:2]
        dimensions_changed = (original_width, original_height) != (edited_width, edited_height)

        # Compare in original-image coordinates.  The original files are never
        # modified; a temporary in-memory normalized copy is used instead.
        resized = cv2.resize(edited, (original_width, original_height), interpolation=cv2.INTER_AREA)
        aligned, alignment_score = self._align(original, resized)
        mask, diff = self._difference_mask(original, aligned)
        regions = self._regions(mask, original_width, original_height)
        changed_area_percent = round(float(np.count_nonzero(mask)) * 100 / mask.size, 2)
        modifications = self._describe_changes(
            changed_area_percent, regions, dimensions_changed, original, aligned
        )

        comparison_id = secrets.token_hex(16)
        output_dir = settings.artifact_dir / "comparisons" / comparison_id
        output_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = output_dir / "difference_heatmap.png"
        overlay_path = output_dir / "change_overlay.png"
        self._save_artifacts(original, diff, mask, heatmap_path, overlay_path)

        return {
            "comparison_id": comparison_id,
            "original_dimensions": {"width": original_width, "height": original_height},
            "edited_dimensions": {"width": edited_width, "height": edited_height},
            "dimensions_changed": dimensions_changed,
            "alignment_score": round(alignment_score, 3),
            "changed_area_percent": changed_area_percent,
            "modifications": modifications,
            "changed_regions": regions,
            "artifacts": {
                "difference_heatmap": f"/artifacts/comparisons/{comparison_id}/difference_heatmap.png",
                "change_overlay": f"/artifacts/comparisons/{comparison_id}/change_overlay.png",
            },
            "limitations": [
                "Reports visible pixel changes after alignment; it cannot prove intent or identify every editing tool.",
                "Large crops, perspective changes, or severe recompression can reduce alignment accuracy.",
            ],
        }

    @staticmethod
    def _decode(data: bytes, label: str) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ImageValidationError(f"The {label} file is not a valid supported image")
        return image

    @staticmethod
    def _align(original: np.ndarray, edited: np.ndarray) -> tuple[np.ndarray, float]:
        reference = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        candidate = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            score, warp = cv2.findTransformECC(
                reference, candidate, warp, cv2.MOTION_EUCLIDEAN,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5),
            )
            aligned = cv2.warpAffine(
                edited, warp, (original.shape[1], original.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return aligned, float(score)
        except cv2.error:
            return edited, 0.0

    @staticmethod
    def _difference_mask(original: np.ndarray, edited: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        difference = cv2.absdiff(original, edited)
        gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        # Blur suppresses JPEG/noise-level differences and preserves meaningful edits.
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(smooth, 24, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask, gray

    @staticmethod
    def _regions(mask: np.ndarray, width: int, height: int) -> list[dict]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: list[dict] = []
        min_area = max(64, int(width * height * 0.00015))
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            regions.append({
                "x": int(x), "y": int(y), "width": int(w), "height": int(h),
                "area_percent": round(float(area) * 100 / (width * height), 2),
            })
        return sorted(regions, key=lambda region: region["area_percent"], reverse=True)[:20]

    @staticmethod
    def _describe_changes(
        changed_percent: float, regions: list[dict], dimensions_changed: bool,
        original: np.ndarray, edited: np.ndarray,
    ) -> list[str]:
        modifications: list[str] = []
        if dimensions_changed:
            modifications.append("DIMENSIONS_OR_CROP_CHANGED")
        if changed_percent < 0.05:
            return modifications or ["NO_MATERIAL_VISUAL_CHANGE"]
        if changed_percent > 75:
            original_mean = cv2.mean(original)[:3]
            edited_mean = cv2.mean(edited)[:3]
            if np.mean(np.abs(np.subtract(original_mean, edited_mean))) > 12:
                modifications.append("GLOBAL_COLOR_OR_LIGHTING_ADJUSTMENT")
            else:
                modifications.append("WIDESPREAD_VISUAL_CHANGE")
        elif regions:
            modifications.append("LOCALIZED_CONTENT_CHANGE")
        else:
            modifications.append("VISUAL_CHANGE_DETECTED")
        return modifications

    @staticmethod
    def _save_artifacts(original: np.ndarray, diff: np.ndarray, mask: np.ndarray, heatmap_path: Path, overlay_path: Path) -> None:
        heatmap = cv2.applyColorMap(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET)
        cv2.imwrite(str(heatmap_path), heatmap)
        overlay = original.copy()
        overlay[mask > 0] = (0, 0, 255)
        cv2.imwrite(str(overlay_path), cv2.addWeighted(original, 0.6, overlay, 0.4, 0))
