"""Manipulation dataset preparation."""
import argparse
from pathlib import Path

from training.dataset_adapter import build_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="datasets/manipulation/manifest.csv")
    parser.add_argument("--type", default="casia", choices=["casia", "fau", "generic"])
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: {root} not found")
        return 1

    manifest = build_manifest(root, Path(args.output), dataset_type=args.type, task="manipulation")
    print(f"Manifest: {len(manifest)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
