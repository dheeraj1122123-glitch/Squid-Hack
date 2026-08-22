"""Dataset download helper script."""
import argparse
import json
from pathlib import Path

from app.core.config import DATASET_LINKS, settings


DATASET_INFO = {
    "dresden": {
        "url": "https://www.mmlab.ie.cuhk.edu.hk/archive/fh/camera_model.htm",
        "description": "Dresden Image Database — 25 camera models, native and JPEG images",
        "local_dir": "datasets/camera/dresden",
        "type": "camera",
        "manual": True,
        "instructions": "Register and download from the CUHK page. Extract to datasets/camera/dresden/",
    },
    "nist_mfc": {
        "url": "https://mfc.nist.gov/",
        "description": "NIST Media Forensics Challenge datasets",
        "local_dir": "datasets/camera/nist_mfc",
        "type": "camera",
        "manual": True,
        "instructions": "Register at mfc.nist.gov and download challenge datasets.",
    },
    "casia2": {
        "url": "http://forensics.idealtest.org/digitalimage/CASIA2.0/CASIA2.0.html",
        "description": "CASIA 2.0 Image Tampering Detection Evaluation Database",
        "local_dir": "datasets/manipulation/casia2",
        "type": "manipulation",
        "manual": True,
        "instructions": "Request access from CASIA. Extract to datasets/manipulation/casia2/",
    },
    "casia_groundtruth": {
        "url": "https://github.com/namtpham/casia2groundtruth",
        "description": "CASIA 2.0 ground truth masks",
        "local_dir": "datasets/manipulation/casia2_groundtruth",
        "type": "manipulation",
        "manual": False,
        "git": "https://github.com/namtpham/casia2groundtruth.git",
    },
    "fau": {
        "url": "https://www5.cs.fau.de/research/data/image-manipulation/",
        "description": "FAU Image Manipulation Dataset",
        "local_dir": "datasets/manipulation/fau",
        "type": "manipulation",
        "manual": True,
        "instructions": "Download from FAU research page. Extract to datasets/manipulation/fau/",
    },
    "vision": {
        "url": "http://staff.utia.cas.cz/novozada/db/",
        "description": "UTIA Vision Lab Forensic Datasets",
        "local_dir": "datasets/camera/vision",
        "type": "camera",
        "manual": True,
        "instructions": "Download from UTIA page. Extract to datasets/camera/vision/",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Dataset download instructions and setup")
    parser.add_argument("--list", action="store_true", help="List all datasets")
    parser.add_argument("--clone-groundtruth", action="store_true", help="Clone CASIA ground truth repo")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent / "datasets"
    base.mkdir(parents=True, exist_ok=True)

    if args.list or not args.clone_groundtruth:
        print("=" * 60)
        print("CameraTrace Dataset Configuration")
        print("=" * 60)
        for name, info in DATASET_INFO.items():
            local = base.parent / info["local_dir"] if not str(info["local_dir"]).startswith("datasets") else Path(__file__).resolve().parent.parent / info["local_dir"]
            exists = local.exists() and any(local.rglob("*"))
            status = "FOUND" if exists else "MISSING"
            print(f"\n[{status}] {name}")
            print(f"  URL: {info['url']}")
            print(f"  Dir: {info['local_dir']}")
            print(f"  {info['description']}")
            if info.get("manual"):
                print(f"  -> {info['instructions']}")

        config = {"dataset_links": DATASET_LINKS, "datasets": DATASET_INFO}
        (base / "dataset_config.json").write_text(json.dumps(config, indent=2))
        print(f"\nConfig saved to {base / 'dataset_config.json'}")

    if args.clone_groundtruth:
        import subprocess
        gt_dir = Path(__file__).resolve().parent.parent / "datasets/manipulation/casia2_groundtruth"
        if not gt_dir.exists():
            subprocess.run(["git", "clone", DATASET_INFO["casia_groundtruth"]["git"], str(gt_dir)], check=False)
            print(f"Cloned ground truth to {gt_dir}")
        else:
            print(f"Ground truth already exists at {gt_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
