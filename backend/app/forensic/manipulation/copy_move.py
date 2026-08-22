"""Copy-move forgery detection baseline."""
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _get_detector():
    try:
        return cv2.SIFT_create()
    except (cv2.error, AttributeError):
        return cv2.ORB_create(nfeatures=2000)


def detect_copy_move(img: np.ndarray, min_matches: int = 10) -> dict[str, Any]:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    h, w = gray.shape
    if h < 128 or w < 128:
        return {
            "status": "insufficient_signal",
            "copy_move_detected": False,
            "confidence": 0.0,
            "availability": True,
        }

    detector = _get_detector()
    kp, des = detector.detectAndCompute(gray, None)

    if des is None or len(kp) < min_matches * 2:
        return {
            "status": "success",
            "copy_move_detected": False,
            "confidence": 0.0,
            "matched_regions": [],
            "availability": True,
        }

    bf = cv2.BFMatcher(cv2.NORM_L2 if isinstance(detector, cv2.SIFT) else cv2.NORM_HAMMING)
    matches = bf.knnMatch(des, des, k=3)

    good = []
    pts1, pts2 = [], []
    for m_list in matches:
        if len(m_list) < 2:
            continue
        m, n = m_list[0], m_list[1]
        if m.queryIdx != m.trainIdx and m.distance < 0.75 * n.distance:
            pt1 = kp[m.queryIdx].pt
            pt2 = kp[m.trainIdx].pt
            dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            if dist > 30:
                good.append((pt1, pt2, m.distance))
                pts1.append(pt1)
                pts2.append(pt2)

    inliers_count = 0
    if len(pts1) >= 4:
        p1 = np.float32(pts1).reshape(-1, 1, 2)
        p2 = np.float32(pts2).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        if mask is not None:
            inliers_count = int(np.sum(mask))

    detected = len(good) >= min_matches and inliers_count >= 4
    confidence = min(1.0, inliers_count / (min_matches * 2)) if detected else 0.0

    heatmap = np.zeros((h, w), dtype=np.float32)
    regions = []
    for pt1, pt2, _ in good[:50]:
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        cv2.circle(heatmap, (x1, y1), 8, 1.0, -1)
        cv2.circle(heatmap, (x2, y2), 8, 1.0, -1)
        regions.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return {
        "status": "success",
        "copy_move_detected": detected,
        "confidence": confidence,
        "match_count": len(good),
        "ransac_inliers": inliers_count,
        "matched_regions": regions[:20],
        "heatmap": heatmap,
        "availability": True,
        "limitations": ["RANSAC verified baseline; extreme warping or heavy compression may impact detection"],
    }

