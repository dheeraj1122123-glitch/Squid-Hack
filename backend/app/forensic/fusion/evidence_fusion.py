"""Evidence fusion from forensic modules."""
from typing import Any


def fuse_module_evidence(modules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build transparent evidence table from module outputs."""
    table: list[dict] = []
    for name, result in modules.items():
        table.append({
            "module": name,
            "status": result.get("status", "unknown"),
            "score": result.get("score") or result.get("confidence") or result.get("local_noise_anomaly_score"),
            "availability": result.get("availability", True),
            "evidence": {k: v for k, v in result.items() if k not in ("residual", "ela_map", "anomaly_map", "heatmap")},
        })
    available = [t for t in table if t["availability"]]
    return {
        "evidence_table": table,
        "modules_available": len(available),
        "modules_total": len(table),
    }
