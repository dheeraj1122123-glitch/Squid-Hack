"""Train script for AI-generated image detection module."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from app.core.config import settings
from app.forensic.manipulation.ai_generator_detector import AIGeneratedClassifier
from app.forensic.preprocessing.image_loader import load_image_rgb


class AIDataset(Dataset):
    def __init__(self, manifest_path: Path, target_size: tuple[int, int] = (256, 256)):
        df = pd.read_csv(manifest_path)
        self.samples = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
            p = Path(row["image_path"])
            label = 1.0 if str(row.get("manipulation_type", "")).upper() == "AI_GENERATED" else 0.0
            if not p.exists():
                continue
            try:
                img = load_image_rgb(p)
                resized = cv2.resize(img, target_size) / 255.0
                t = torch.from_numpy(resized.transpose(2, 0, 1)).float()
                self.samples.append((t, torch.tensor([label], dtype=torch.float32)))
            except Exception:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/manipulation/train.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default="models/manipulation/ai_generator_detector.pt")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest missing: {manifest_path}")
        return 1

    ds = AIDataset(manifest_path)
    if len(ds) == 0:
        print("No samples found.")
        return 1

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = AIGeneratedClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {total_loss / len(ds):.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, out_path)

    registry_path = settings.model_dir / "registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    registry["ai_generator_detector"] = {
        "path": str(out_path),
        "version": "1.0.0",
        "trained": True,
    }
    registry_path.write_text(json.dumps(registry, indent=2))
    print(f"Saved AI generator model to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
