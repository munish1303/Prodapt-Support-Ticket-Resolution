"""Shared fixtures. Unit tests use a deterministic hashing embedder so they run
fast and offline; tests marked `models` load the real HF models."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import app  # noqa: F401  (sets HF cache env)
from app.models.schemas import Document, QueryMetadata
from app.utils.text import content_tokens


class FakeEmbedder:
    """Bag-of-stemmed-words hashed into 384 dims, L2-normalised."""

    dimension = 384

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dimension, dtype=np.float32)
        for tok in content_tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dimension] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts, batch_size: int = 32, show_progress: bool = False):
        if isinstance(texts, str):
            return self._one(texts)
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dimension), dtype=np.float32)


class FakeNLI:
    """Entails when most hypothesis tokens appear in the premise; contradicts on explicit negation."""

    labels = ["contradiction", "entailment", "neutral"]

    def predict(self, pairs):
        out = []
        for premise, hypothesis in pairs:
            h = set(content_tokens(hypothesis))
            p = set(content_tokens(premise))
            overlap = len(h & p) / len(h) if h else 0.0
            negated = any(w in hypothesis.lower().split() for w in ("not", "never", "disable", "don't"))
            if negated and overlap > 0.3:
                out.append({"contradiction": 0.9, "entailment": 0.05, "neutral": 0.05})
            elif overlap >= 0.6:
                out.append({"contradiction": 0.02, "entailment": 0.9, "neutral": 0.08})
            else:
                out.append({"contradiction": 0.05, "entailment": 0.1, "neutral": 0.85})
        return out


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_nli() -> FakeNLI:
    return FakeNLI()


@pytest.fixture
def metadata() -> QueryMetadata:
    return QueryMetadata(
        intent="connectivity_issue",
        intent_confidence=0.9,
        products=["router"],
        severity="high",
        sentiment="negative",
        sentiment_score=-0.8,
    )


@pytest.fixture
def sources() -> list[Document]:
    return [
        Document(
            "TKT-1",
            "Complaint: wifi drops every evening.\nResolution: Investigated wifi dropouts. "
            "Run the router's channel scan and move the 2.4 GHz radio to the least congested channel. "
            "Enable the 5 GHz band and connect nearby devices to it. Issue resolved and customer confirmed service is working.",
            {"type": "ticket", "category": "connectivity_issue", "resolved": True},
        ),
        Document(
            "KB-1",
            "Fixing evening Wi-Fi dropouts\nCause: interference.\nResolution steps:\n"
            "1. Run the router's channel scan and move the 2.4 GHz radio to the least congested channel.\n"
            "2. Enable the 5 GHz band and connect nearby devices to it.\n"
            "3. Update the router firmware to the latest version.",
            {"type": "kb_article", "title": "Fixing evening Wi-Fi dropouts"},
        ),
        Document(
            "TKT-2",
            "Complaint: wifi cuts out at night.\nResolution: Agent action: Enable the 5 GHz band and connect "
            "nearby devices to it. Update the router firmware to the latest version. Resolved on first contact.",
            {"type": "ticket", "category": "connectivity_issue", "resolved": True},
        ),
    ]
