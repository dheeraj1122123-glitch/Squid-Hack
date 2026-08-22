"""Camera dataset preparation."""
import argparse
from pathlib import Path

from training.dataset_adapter import build_manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare camera dataset manifest")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--output", type=str, default="datasets/camera/manifest.csv")
    parser.add_argument("--type", type=str, default="dresden", choices=["dresden", "nist_mfc", "generic"])
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: Directory not found: {root}")
        print("Download datasets first: python scripts/download_datasets.py")
        return 1

    manifest = build_manifest(root, Path(args.output), dataset_type=args.type, task="camera")
    print(f"Created manifest with {len(manifest)} images -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
