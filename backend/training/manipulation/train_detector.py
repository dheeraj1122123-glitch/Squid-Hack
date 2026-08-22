"""Train manipulation detector."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from app.forensic.manipulation.copy_move import detect_copy_move
from app.forensic.manipulation.jpeg_analysis import analyze_jpeg_manipulation
from app.forensic.manipulation.noise_inconsistency import local_noise_inconsistency
from app.forensic.manipulation.splice import detect_splicing
from app.forensic.preprocessing.image_loader import load_image_rgb
from app.core.config import settings


def extract_manip_features(path: Path) -> np.ndarray:
    img = load_image_rgb(path)
    noise = local_noise_inconsistency(img)
    splice = detect_splicing(img, path)
    jpeg = analyze_jpeg_manipulation(img, path)
    cm = detect_copy_move(img)
    return np.array([
        noise.get("local_noise_anomaly_score", 0),
        splice.get("tampered_probability", 0),
        jpeg.get("compression_inconsistency_score", 0),
        cm.get("confidence", 0),
        float(cm.get("copy_move_detected", False)),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/manipulation/manifest.csv")
    parser.add_argument("--output-dir", default="models/manipulation")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = Path(row["image_path"])
        if not path.exists():
            continue
        try:
            X.append(extract_manip_features(path))
            label = row.get("manipulation_type", "AUTHENTIC")
            y.append("MANIPULATED" if label != "AUTHENTIC" else "AUTHENTIC")
        except Exception:
            continue

    if len(X) < 20:
        print(f"Insufficient samples ({len(X)}). Need at least 20.")
        return 1

    X = np.array(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / "manipulation_detector.joblib")

    registry_path = settings.model_dir / "registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    registry["manipulation_detector"] = {"path": str(out / "manipulation_detector.joblib"), "trained": True, "accuracy": acc}
    registry_path.write_text(json.dumps(registry, indent=2))
    print(f"Trained on {len(X)} samples, test accuracy: {acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
