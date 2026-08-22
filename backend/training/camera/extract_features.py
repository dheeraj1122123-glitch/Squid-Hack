"""Extract forensic features from camera dataset."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from app.forensic.camera.cfa import extract_cfa_features
from app.forensic.camera.features import build_feature_vector
from app.forensic.camera.frequency import frequency_analysis
from app.forensic.camera.jpeg import analyze_compression
from app.forensic.camera.noise import analyze_noise
from app.forensic.camera.prnu import analyze_prnu
from app.forensic.preprocessing.image_loader import load_image_rgb


def extract_features_for_image(path: Path) -> tuple[np.ndarray, list[str]]:
    img = load_image_rgb(path)
    noise = analyze_noise(img)
    cfa = extract_cfa_features(img)
    freq = frequency_analysis(img)
    compression = analyze_compression(img, path)
    prnu = analyze_prnu(img)
    noise_clean = {k: v for k, v in noise.items() if k != "residual"}
    return build_feature_vector(noise_clean, cfa, freq, compression, prnu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/camera/train.csv")
    parser.add_argument("--output", default="datasets/camera/features.npz")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    features_list = []
    labels = []
    feature_names = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        path = Path(row["image_path"])
        if not path.exists():
            continue
        try:
            vec, names = extract_features_for_image(path)
            if feature_names is None:
                feature_names = names
            features_list.append(vec)
            labels.append(row.get("camera_model", "unknown"))
        except Exception as e:
            print(f"Skip {path}: {e}")

    if not features_list:
        print("No features extracted. Check dataset paths.")
        return 1

    X = np.vstack(features_list)
    np.savez(args.output, X=X, y=np.array(labels), feature_names=feature_names)
    schema = {"feature_names": feature_names, "n_samples": len(labels), "n_features": X.shape[1]}
    Path(args.output).with_suffix(".schema.json").write_text(json.dumps(schema, indent=2))
    print(f"Saved {X.shape[0]} samples, {X.shape[1]} features -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
