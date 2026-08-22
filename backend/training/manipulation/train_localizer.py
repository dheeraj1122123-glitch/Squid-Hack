"""Train manipulation localizer (placeholder interface)."""
import argparse
import json
from pathlib import Path

from app.core.config import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/manipulation/manifest.csv")
    args = parser.parse_args()

    print("Manipulation localizer requires pixel-level ground truth masks.")
    print("Interface ready. Train when CASIA/FAU datasets with masks are available.")

    registry_path = settings.model_dir / "registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    registry["manipulation_localizer"] = {"trained": False, "status": "requires_ground_truth_masks"}
    registry_path.write_text(json.dumps(registry, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
