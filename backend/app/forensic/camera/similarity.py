"""Camera similarity search."""
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class SimilaritySearch:
    def __init__(self, index_dir: Path | None = None):
        self.index_dir = index_dir or settings.model_dir / "camera"
        self.embeddings: np.ndarray | None = None
        self.labels: list[str] = []
        self.index = None
        self._load()

    def _load(self) -> None:
        emb_path = self.index_dir / "embeddings.npy"
        labels_path = self.index_dir / "embedding_labels.json"
        if emb_path.exists() and labels_path.exists():
            self.embeddings = np.load(emb_path)
            self.labels = json.loads(labels_path.read_text())
            if FAISS_AVAILABLE and len(self.embeddings) > 0:
                dim = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dim)
                norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                normalized = self.embeddings / norms
                self.index.add(normalized.astype(np.float32))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        if self.embeddings is None or len(self.labels) == 0:
            return []

        q = query_vector.astype(np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        if FAISS_AVAILABLE and self.index is not None:
            sims, idxs = self.index.search(q.reshape(1, -1).astype(np.float32), top_k)
            results = []
            for rank, (sim, idx) in enumerate(zip(sims[0], idxs[0])):
                if idx >= 0:
                    results.append({
                        "camera_model": self.labels[idx],
                        "similarity": float(sim),
                        "rank": rank + 1,
                    })
            return results

        sims = self.embeddings @ q
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            {
                "camera_model": self.labels[i],
                "similarity": float(sims[i]),
                "rank": rank + 1,
            }
            for rank, i in enumerate(top_idx)
        ]
