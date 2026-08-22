"""Evidence summary builder."""
from typing import Any


def build_evidence_summary(fused: dict[str, Any]) -> dict[str, Any]:
    table = fused.get("evidence_table", [])
    summary = {
        "total_modules": fused.get("modules_total", 0),
        "available_modules": fused.get("modules_available", 0),
        "module_summaries": [],
    }
    for entry in table:
        summary["module_summaries"].append({
            "module": entry["module"],
            "status": entry["status"],
            "available": entry["availability"],
        })
    return summary
