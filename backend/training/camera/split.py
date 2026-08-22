"""Train/test split with leakage prevention."""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/camera/manifest.csv")
    parser.add_argument("--output-dir", default="datasets/camera")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-by", default="scene", choices=["scene", "device", "image"])
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.split_by == "device" and "physical_device" in df.columns:
        groups = df["physical_device"].fillna(df.index.astype(str))
    elif args.split_by == "scene" and "scene" in df.columns:
        groups = df["scene"].fillna(df["camera_model"].fillna(df.index.astype(str)))
    else:
        groups = df.index

    if "camera_model" in df.columns:
        groups = df["camera_model"].astype(str) + "_" + groups.astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    meta = {
        "split_by": args.split_by,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "classes": sorted(df["camera_model"].dropna().unique().tolist()) if "camera_model" in df.columns else [],
    }
    (output_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
