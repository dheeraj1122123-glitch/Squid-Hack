"""Generate demo data for development testing."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_demo_camera_data():
    base = Path(__file__).resolve().parent.parent / "datasets" / "camera" / "demo"
    cameras = [
        ("Canon", "EOS_90D"),
        ("Sony", "A6400"),
        ("Nikon", "D7500"),
    ]
    for make, model in cameras:
        cam_dir = base / make / model
        cam_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            rng = np.random.RandomState(hash(f"{make}{model}{i}") % 2**31)
            img = rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            noise = rng.normal(0, 2 + i * 0.5, img.shape)
            img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(cam_dir / f"img_{i:03d}.jpg"), img)
    print(f"Generated demo camera data in {base}")
    return base


def generate_demo_manipulation_data():
    base = Path(__file__).resolve().parent.parent / "datasets" / "manipulation" / "demo"
    auth_dir = base / "authentic"
    tamper_dir = base / "tampered"
    auth_dir.mkdir(parents=True, exist_ok=True)
    tamper_dir.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        rng = np.random.RandomState(i)
        img = rng.randint(50, 200, (400, 600, 3), dtype=np.uint8)
        cv2.imwrite(str(auth_dir / f"auth_{i:03d}.jpg"), img)

        tampered = img.copy()
        tampered[100:200, 200:400] = rng.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(tamper_dir / f"tamp_{i:03d}.jpg"), tampered)

    print(f"Generated demo manipulation data in {base}")
    return base


def main():
    generate_demo_camera_data()
    generate_demo_manipulation_data()
    print("\nTo prepare manifests:")
    print("  python -m training.camera.prepare_dataset --root datasets/camera/demo --type generic")
    print("  python -m training.manipulation.prepare_dataset --root datasets/manipulation/demo --type generic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
