"""Train camera model classifier."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from app.core.config import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="datasets/camera/features.npz")
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "random_forest", "svm"])
    parser.add_argument("--output-dir", default="models/camera")
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X, y = data["X"], data["y"]
    feature_names = data["feature_names"].tolist()

    if len(X) < 10:
        print(f"Insufficient samples ({len(X)}). Need at least 10 for training.")
        return 1

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if args.model == "xgboost":
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42,
        )
    elif args.model == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    else:
        from sklearn.svm import SVC
        clf = SVC(probability=True, kernel="rbf", random_state=42)

    clf = CalibratedClassifierCV(clf, cv=min(3, len(set(y_enc))))
    clf.fit(X_scaled, y_enc)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / "model.joblib")
    joblib.dump(scaler, out / "scaler.joblib")
    joblib.dump(le, out / "label_encoder.joblib")
    (out / "feature_schema.json").write_text(json.dumps({"feature_names": feature_names}, indent=2))

    registry_path = settings.model_dir / "registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    registry["camera_classifier"] = {
        "path": str(out / "model.joblib"),
        "version": "1.0.0",
        "feature_schema": str(out / "feature_schema.json"),
        "trained": True,
        "model_type": args.model,
        "n_classes": len(le.classes_),
    }
    registry_path.write_text(json.dumps(registry, indent=2))
    print(f"Trained {args.model} on {len(X)} samples, {len(le.classes_)} classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
