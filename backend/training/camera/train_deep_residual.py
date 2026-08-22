"""Training pipeline for PyTorch Deep Residual Patch model."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from app.core.config import settings
from app.forensic.camera.deep_residual_model import CameraResidualCNN
from app.forensic.camera.noise import extract_noise_residual
from app.forensic.preprocessing.image_loader import load_image_rgb
from app.forensic.preprocessing.patching import extract_patches


class ResidualPatchDataset(Dataset):
    def __init__(self, manifest_path: Path, patch_size: int = 128):
        df = pd.read_csv(manifest_path)
        self.patch_size = patch_size
        self.samples = []
        self.classes = sorted(df["camera_model"].dropna().unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting patches"):
            img_path = Path(row["image_path"])
            label = row.get("camera_model")
            if not img_path.exists() or label not in self.class_to_idx:
                continue
            idx = self.class_to_idx[label]
            try:
                img = load_image_rgb(img_path)
                residual = extract_noise_residual(img, method="gaussian")
                for p, _ in extract_patches(residual, patch_size=patch_size, stride=patch_size):
                    if p.ndim == 2:
                        p = np.stack([p] * 3, axis=-1)
                    p_tensor = torch.from_numpy(p.transpose(2, 0, 1)).float()
                    self.samples.append((p_tensor, idx))
            except Exception:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/camera/train.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", default="models/camera/camera_residual_model.pt")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest missing: {manifest_path}")
        return 1

    ds = ResidualPatchDataset(manifest_path)
    if len(ds) == 0:
        print("No residual patches extracted.")
        return 1

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = CameraResidualCNN(num_classes=len(ds.classes))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
        acc = correct / max(1, total)
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {total_loss / total:.4f} - Acc: {acc:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "classes": ds.classes,
        "num_classes": len(ds.classes),
    }
    torch.save(checkpoint, out_path)

    registry_path = settings.model_dir / "registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    registry["deep_camera_residual"] = {
        "path": str(out_path),
        "version": "1.0.0",
        "trained": True,
        "classes": ds.classes,
    }
    registry_path.write_text(json.dumps(registry, indent=2))
    print(f"Saved model to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
