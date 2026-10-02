"""Sentence-embedding service (all-MiniLM-L6-v2 by default, 384-dim, normalised)."""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from app.config import settings


class EmbeddingService:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name, device="cpu")
        self.dimension = self.model.get_embedding_dimension()
        self._lock = threading.Lock()  # torch inference is not re-entrant across threads

    def encode(self, texts: str | list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        with self._lock:
            return self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )


@lru_cache(maxsize=4)
def get_embedding_service(model_name: str | None = None) -> EmbeddingService:
    return EmbeddingService(model_name)


def to_pgvector(vec: np.ndarray | list[float]) -> str:
    """Serialise a vector to pgvector's text format ('[0.1,0.2,...]')."""
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"
