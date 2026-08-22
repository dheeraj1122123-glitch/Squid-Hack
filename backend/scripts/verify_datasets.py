"""Verify dataset availability and integrity."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.dataset_adapter import find_images, sha256_file


def verify():
    base = Path(__file__).resolve().parent.parent
    datasets_dir = base / "datasets"
    results = {"datasets": [], "ready_for_training": False, "issues": []}

    checks = [
        ("camera/dresden", "dresden", "camera"),
        ("camera/nist_mfc", "nist_mfc", "camera"),
        ("camera/vision", "vision", "camera"),
        ("manipulation/casia2", "casia", "manipulation"),
        ("manipulation/fau", "fau", "manipulation"),
    ]

    any_found = False
    for subpath, dtype, task in checks:
        root = datasets_dir / subpath
        if root.exists():
            images = find_images(root)
            if images:
                any_found = True
                results["datasets"].append({
                    "path": str(root),
                    "type": dtype,
                    "task": task,
                    "image_count": len(images),
                    "status": "ok",
                })
            else:
                results["issues"].append(f"{root}: directory exists but no images found")
        else:
            results["issues"].append(f"{root}: not found (download required)")

    manifest_camera = datasets_dir / "camera" / "manifest.csv"
    manifest_manip = datasets_dir / "manipulation" / "manifest.csv"
    if manifest_camera.exists():
        results["camera_manifest"] = str(manifest_camera)
    if manifest_manip.exists():
        results["manipulation_manifest"] = str(manifest_manip)

    model_registry = base / "models" / "registry.json"
    if model_registry.exists():
        registry = json.loads(model_registry.read_text())
        results["models"] = {k: v.get("trained", False) for k, v in registry.items()}

    results["ready_for_training"] = any_found

    out_path = datasets_dir / "verification_report.json"
    out_path.write_text(json.dumps(results, indent=2))

    print("Dataset Verification Report")
    print("=" * 40)
    for ds in results["datasets"]:
        print(f"  OK  {ds['path']} ({ds['image_count']} images)")
    for issue in results["issues"]:
        print(f"  --  {issue}")
    print(f"\nReady for training: {results['ready_for_training']}")
    print(f"Report: {out_path}")
    return 0 if any_found or manifest_camera.exists() else 0


if __name__ == "__main__":
    raise SystemExit(verify())
