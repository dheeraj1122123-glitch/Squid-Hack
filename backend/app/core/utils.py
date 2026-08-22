"""Utility functions for CameraTrace core."""
from typing import Any
import numpy as np


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types and non-serializable objects to standard Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    return obj
