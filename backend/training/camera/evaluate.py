"""Evaluate camera classifier."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from training.camera.extract_features import extract_features_for_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-manifest", default="datasets/camera/test.csv")
    parser.add_argument("--model-dir", default="models/camera")
    parser.add_argument("--output", default="datasets/camera/evaluation.json")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not (model_dir / "model.joblib").exists():
        print("Model not trained. Run: python -m training.camera.train_classifier")
        return 1

    clf = joblib.load(model_dir / "model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    le = joblib.load(model_dir / "label_encoder.joblib")

    df = pd.read_csv(args.test_manifest)
    y_true, y_pred, y_proba_all = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = Path(row["image_path"])
        if not path.exists():
            continue
        try:
            vec, _ = extract_features_for_image(path)
            X = scaler.transform(vec.reshape(1, -1))
            pred = clf.predict(X)[0]
            proba = clf.predict_proba(X)[0]
            y_true.append(row["camera_model"])
            y_pred.append(le.inverse_transform([pred])[0])
            y_proba_all.append(proba)
        except Exception:
            continue

    if not y_true:
        print("No test samples evaluated.")
        return 1

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    top1 = accuracy_score(y_true_arr, y_pred_arr)
    top3 = top1
    if y_proba_all:
        top3_count = 0
        for i, proba in enumerate(y_proba_all):
            top3_idx = np.argsort(proba)[::-1][:3]
            top3_labels = le.inverse_transform(top3_idx)
            if y_true[i] in top3_labels:
                top3_count += 1
        top3 = top3_count / len(y_true)

    report = {
        "accuracy": float(top1),
        "top1_accuracy": float(top1),
        "top3_accuracy": float(top3),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr).tolist(),
        "classes": le.classes_.tolist(),
        "classification_report": classification_report(y_true_arr, y_pred_arr, zero_division=0),
        "n_test": len(y_true),
    }

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"Top-1 Accuracy: {top1:.4f}, Top-3: {top3:.4f}, Macro F1: {report['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
