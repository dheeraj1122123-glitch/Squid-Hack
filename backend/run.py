"""Entry point for CameraTrace backend platform."""
import os
import sys
import webbrowser
from pathlib import Path

import uvicorn

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def auto_setup() -> None:
    """Auto-setup demo data and train initial baseline camera classifier if not present."""
    model_file = BASE_DIR / "models" / "camera" / "model.joblib"
    if not model_file.exists():
        print("=" * 60)
        print("CameraTrace: First-Time Setup & Baseline Model Training...")
        print("=" * 60)
        try:
            from scripts.generate_demo_data import generate_demo_camera_data, generate_demo_manipulation_data
            from training.dataset_adapter import build_manifest
            
            # 1. Generate demo datasets
            cam_demo = BASE_DIR / "datasets" / "camera" / "demo"
            manip_demo = BASE_DIR / "datasets" / "manipulation" / "demo"
            if not cam_demo.exists():
                generate_demo_camera_data()
            if not manip_demo.exists():
                generate_demo_manipulation_data()

            # 2. Build manifests
            cam_manifest = BASE_DIR / "datasets" / "camera" / "manifest.csv"
            if not cam_manifest.exists():
                build_manifest(cam_demo, cam_manifest, dataset_type="generic", task="camera")

            # 3. Extract features & train model
            from training.camera.extract_features import main as extract_main
            from training.camera.train_classifier import main as train_main
            
            sys.argv = ["extract_features", "--manifest", str(cam_manifest)]
            extract_main()

            features_npz = BASE_DIR / "datasets" / "camera" / "features.npz"
            sys.argv = ["train_classifier", "--features", str(features_npz), "--model", "xgboost"]
            train_main()
            print("Auto-setup complete! Baseline model ready.")
        except Exception as e:
            print(f"Auto-setup warning: {e}. Platform will run with model state MODEL_NOT_TRAINED.")


if __name__ == "__main__":
    auto_setup()
    
    url = "http://127.0.0.1:8000/"
    print("\n" + "=" * 60)
    print(f"CameraTrace Web UI & API Server starting on: {url}")
    print("=" * 60 + "\n")

    # Open browser automatically after a short delay
    def open_browser():
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import threading
    threading.Timer(1.5, open_browser).start()

    from app.core.config import settings
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
