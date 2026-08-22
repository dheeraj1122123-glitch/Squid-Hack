"""Evaluate manipulation detector."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from training.manipulation.train_detector import extract_manip_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/manipulation/manifest.csv")
    parser.add_argument("--model", default="models/manipulation/manipulation_detector.joblib")
    parser.add_argument("--output", default="datasets/manipulation/evaluation.json")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print("Model not trained.")
        return 1

    clf = joblib.load(args.model)
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

    if not X:
        return 1

    X = np.array(X)
    y_pred = clf.predict(X)
    report = {
        "classification_report": classification_report(y, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "n_samples": len(y),
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(report["classification_report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
