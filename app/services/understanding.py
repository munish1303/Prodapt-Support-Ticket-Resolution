"""Query understanding: intent, products, severity and sentiment (plan §16).

Design notes
------------
* Intent (default "knn"): similarity-weighted k-NN vote over labelled historical
  tickets (complaint-only embeddings, queried through pgvector) plus the
  taxonomy's example phrases. New ticket classes become recognisable as soon as
  their tickets are ingested; no retraining step. Low vote share or low nearest
  similarity => `unknown_intent` (routes to a human). An LLM classifier is
  available via INTENT_CLASSIFIER=llm.
* Severity: base level from the neighbours' severity labels, adjusted by
  explicit urgency/business-impact cues in the complaint (rules), optionally
  bumped by strongly negative sentiment (plan §16.5).
* Products: dictionary/alias matching on word boundaries (optional LLM fallback).
* Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest; score = p(pos) - p(neg).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from sqlalchemy import text

from app.config import settings
from app.core.llm import LLMProvider, LLMUnavailable
from app.core.prompts import (
    INTENT_SYSTEM_PROMPT,
    PRODUCT_SYSTEM_PROMPT,
    build_intent_user_prompt,
)
from app.models.embeddings import EmbeddingService, to_pgvector
from app.models.schemas import QueryMetadata
from app.utils.text import sanitize_for_prompt

logger = logging.getLogger(__name__)

UNKNOWN_INTENT = "unknown_intent"
SEVERITY_LEVELS = ["low", "medium", "high", "critical"]


# --------------------------------------------------------------------------- taxonomy
@dataclass
class IntentDef:
    intent_name: str
    description: str
    examples: list[str]


@dataclass
class ProductDef:
    product_name: str
    category: str
    aliases: list[str]


def load_taxonomy_files(directory: Path | None = None) -> tuple[list[IntentDef], list[ProductDef]]:
    directory = directory or settings.TAXONOMY_DIR
    intents = [
        IntentDef(**{k: d[k] for k in ("intent_name", "description", "examples")})
        for d in json.loads((directory / "intents.json").read_text(encoding="utf-8"))
    ]
    products = [ProductDef(**d) for d in json.loads((directory / "products.json").read_text(encoding="utf-8"))]
    return intents, products


async def load_taxonomy_db(session) -> tuple[list[IntentDef], list[ProductDef]]:
    rows = (
        await session.execute(
            text(
                "SELECT intent_name, coalesce(description,''), examples FROM intent_taxonomy WHERE active ORDER BY intent_id"
            )
        )
    ).all()
    intents = [IntentDef(r[0], r[1], list(r[2] or [])) for r in rows]
    prows = (
        await session.execute(
            text(
                "SELECT product_name, coalesce(category,''), coalesce(aliases, '{}') FROM product_taxonomy WHERE active ORDER BY product_id"
            )
        )
    ).all()
    products = [ProductDef(r[0], r[1], list(r[2])) for r in prows]
    return intents, products


# --------------------------------------------------------------------------- products
class ProductExtractor:
    def __init__(self, products: list[ProductDef]):
        self.set_products(products)

    def set_products(self, products: list[ProductDef]) -> None:
        self._patterns = []
        for p in products:
            terms = sorted({p.product_name.replace("_", " "), *p.aliases}, key=len, reverse=True)
            pattern = re.compile(r"(?<![\w-])(" + "|".join(re.escape(t.lower()) for t in terms) + r")(?![\w-])")
            self._patterns.append((p.product_name, pattern))

    def extract(self, complaint: str) -> list[str]:
        lowered = complaint.lower()
        hits = []
        for name, pattern in self._patterns:
            match = pattern.search(lowered)
            if match:
                hits.append((match.start(), name))
        return [name for _, name in sorted(hits)]


# --------------------------------------------------------------------------- severity
CRITICAL_CUES = [
    r"no (?:internet|service|signal|connection) at all",
    r"completely (?:down|dead|offline)",
    r"emergency",
    r"\burgent\b",
    r"can'?t trade",
    r"whole (?:street|neighbou?rhood|estate|area|block|village|town)",
    r"everyone (?:around|in)",
    r"nobody (?:on|in)",
    r"cable .*(?:down|hanging)",
]
BUSINESS_CUES = [
    r"\bbusiness\b",
    r"work(?:ing)? from home",
    r"\bclients?\b",
    r"card payments",
    r"\bshop\b",
    r"costing me (?:money|business)",
    r"\bcustomers\b",
]
LOW_CUES = [r"\bhow (?:do|can) i\b", r"\bwondering\b", r"\bquestion\b", r"\bcurious\b", r"\bno rush\b", r"\bminor\b"]


def _matches(patterns: list[str], text_: str) -> bool:
    return any(re.search(p, text_) for p in patterns)


def bump_severity(level: str, steps: int = 1) -> str:
    idx = max(0, min(SEVERITY_LEVELS.index(level) + steps, len(SEVERITY_LEVELS) - 1))
    return SEVERITY_LEVELS[idx]


def classify_severity(
    complaint: str,
    neighbour_severities: list[tuple[str, float]],
    sentiment_score: float,
    intent_is_unknown: bool,
) -> str:
    """Base severity from similar tickets, adjusted by explicit cues in this complaint."""
    lowered = complaint.lower()
    # Neighbour labels already include their own business-impact bumps, so the
    # lower median is used as the "intrinsic" severity of this issue type.
    ordered = sorted(SEVERITY_LEVELS.index(s) for s, _ in neighbour_severities if s in SEVERITY_LEVELS)
    base = SEVERITY_LEVELS[ordered[(len(ordered) - 1) // 2]] if ordered else "medium"

    severity = base
    if _matches(CRITICAL_CUES, lowered):
        severity = "critical"
    elif _matches(BUSINESS_CUES, lowered):
        severity = bump_severity(base)
    elif _matches(LOW_CUES, lowered) and base == "medium":
        severity = "low"

    if (
        settings.SEVERITY_NEGATIVE_SENTIMENT_BUMP > -1.0
        and sentiment_score < settings.SEVERITY_NEGATIVE_SENTIMENT_BUMP
        and severity == "medium"
    ):
        severity = "high"
    if intent_is_unknown and severity == "critical" and not _matches(CRITICAL_CUES, lowered):
        severity = "high"  # don't auto-assign critical for unrecognised issues without explicit cues
    return severity


# --------------------------------------------------------------------------- sentiment
class SentimentAnalyzer:
    def __init__(self, model_name: str | None = None):
        from transformers import pipeline

        make_pipeline: Any = pipeline  # transformers' overloads don't cover top_k=None
        self._pipe = make_pipeline(
            "sentiment-analysis", model=model_name or settings.SENTIMENT_MODEL, top_k=None, device=-1
        )

    def analyze(self, complaint: str) -> dict:
        scores = self._pipe(complaint[:1000], truncation=True)
        if scores and isinstance(scores[0], list):
            scores = scores[0]
        probs = {_norm_sentiment_label(s["label"]): float(s["score"]) for s in scores}
        polarity = probs.get("positive", 0.0) - probs.get("negative", 0.0)
        label = label_from_polarity(polarity, probs)
        return {"sentiment": label, "sentiment_score": polarity, "confidence": probs.get(label, 0.0)}


def label_from_polarity(polarity: float, probs: dict[str, float]) -> str:
    """Complaints describe problems, so an off-the-shelf model leans negative even when no
    frustration is expressed. Calibrated polarity thresholds (chosen on the dev split; see
    EVALUATION.md) map to the label definition used here; unset thresholds => argmax."""
    neg_t, pos_t = settings.SENTIMENT_NEGATIVE_THRESHOLD, settings.SENTIMENT_POSITIVE_THRESHOLD
    if neg_t is None or pos_t is None:
        return max(probs, key=lambda label: probs[label])
    if polarity < neg_t:
        return "negative"
    if polarity > pos_t:
        return "positive"
    return "neutral"


def _norm_sentiment_label(label: str) -> str:
    return {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}.get(label.lower(), label.lower())


# --------------------------------------------------------------------------- intent
@dataclass
class Neighbour:
    category: str
    severity: str
    similarity: float
    products: list[str]


class NeighbourIndex(Protocol):
    async def nearest(self, embedding: np.ndarray, k: int) -> list[Neighbour]:
        """Return the k nearest labelled historical tickets."""


class PgNeighbourIndex:
    """k-NN over labelled corpus tickets using pgvector on complaint-only embeddings."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def nearest(self, embedding: np.ndarray, k: int) -> list[Neighbour]:
        # Bounded so a hung database fails fast (TimeoutError) instead of stalling the request.
        return await asyncio.wait_for(self._nearest(embedding, k), timeout=settings.DB_QUERY_TIMEOUT_S)

    async def _nearest(self, embedding: np.ndarray, k: int) -> list[Neighbour]:
        vec = to_pgvector(embedding)
        async with self.session_factory() as session:
            await session.execute(text(f"SET LOCAL ivfflat.probes = {int(settings.IVFFLAT_PROBES)}"))
            rows = (
                await session.execute(
                    text("""
                SELECT category, severity, 1 - (complaint_embedding <=> CAST(:v AS vector)) AS sim,
                       coalesce(metadata->'products', to_jsonb(ARRAY[product])) AS products
                FROM tickets
                WHERE category IS NOT NULL AND complaint_embedding IS NOT NULL
                ORDER BY complaint_embedding <=> CAST(:v AS vector)
                LIMIT :k
                """),
                    {"v": vec, "k": k},
                )
            ).all()
        return [Neighbour(r[0], r[1], float(r[2]), [p for p in (r[3] or []) if p]) for r in rows]


class InMemoryNeighbourIndex:
    """Exact k-NN in numpy; used by tests and offline evaluation."""

    def __init__(
        self,
        embeddings: np.ndarray,
        categories: list[str],
        severities: list[str],
        products: list[list[str]] | None = None,
    ):
        self.embeddings = embeddings
        self.categories = categories
        self.severities = severities
        self.products = products or [[] for _ in categories]

    async def nearest(self, embedding: np.ndarray, k: int) -> list[Neighbour]:
        if len(self.categories) == 0:
            return []
        sims = self.embeddings @ embedding
        idx = np.argsort(-sims)[:k]
        return [Neighbour(self.categories[i], self.severities[i], float(sims[i]), self.products[i]) for i in idx]


def knn_vote(
    neighbours: list[tuple[str, float]],
    min_confidence: float,
    min_similarity: float,
    power: float = 3.0,
) -> dict:
    """Similarity-weighted vote. Weights sim**power so close neighbours dominate."""
    if not neighbours:
        return {
            "intent": UNKNOWN_INTENT,
            "confidence": 0.0,
            "candidates": [],
            "nearest_similarity": 0.0,
            "is_unknown": True,
        }
    weights: dict[str, float] = defaultdict(float)
    for label, sim in neighbours:
        weights[label] += max(sim, 0.0) ** power
    total = sum(weights.values()) or 1.0
    ranked = sorted(((lbl, w / total) for lbl, w in weights.items()), key=lambda x: -x[1])
    top_label, confidence = ranked[0]
    nearest = max(sim for _, sim in neighbours)
    is_unknown = confidence < min_confidence or nearest < min_similarity
    return {
        "intent": UNKNOWN_INTENT if is_unknown else top_label,
        "confidence": float(confidence),
        "candidates": [(lbl, round(float(c), 4)) for lbl, c in ranked[:3]],
        "nearest_similarity": float(nearest),
        "is_unknown": is_unknown,
    }


def infer_products(neighbours: list[Neighbour], min_share: float) -> list[str]:
    """Products implied by the complaint: those present in at least `min_share` of the
    (similarity-weighted) same-class neighbours. Catches implicit mentions such as
    'the web keeps cutting out' (broadband) that dictionary matching cannot."""
    if not neighbours or min_share > 1.0:
        return []
    total = sum(max(n.similarity, 0.0) for n in neighbours) or 1.0
    share: dict[str, float] = defaultdict(float)
    for n in neighbours:
        for p in set(n.products):
            share[p] += max(n.similarity, 0.0) / total
    return [p for p, v in sorted(share.items(), key=lambda x: -x[1]) if v >= min_share]


class UnderstandingService:
    def __init__(
        self,
        embedder: EmbeddingService,
        neighbour_index: NeighbourIndex,
        intents: list[IntentDef],
        products: list[ProductDef],
        sentiment: SentimentAnalyzer | None,
        llm: LLMProvider | None = None,
        classifier: str | None = None,
    ):
        self.embedder = embedder
        self.index = neighbour_index
        self.sentiment = sentiment
        self.llm = llm
        self.classifier = classifier or settings.INTENT_CLASSIFIER
        self.product_extractor = ProductExtractor(products)
        self.set_taxonomy(intents, products)

    def set_taxonomy(self, intents: list[IntentDef], products: list[ProductDef]) -> None:
        self.intents = [i for i in intents if i.intent_name != UNKNOWN_INTENT]
        self.products = products
        self.product_extractor.set_products(products)
        proto_texts, proto_labels = [], []
        for intent in self.intents:
            for ex in intent.examples:
                proto_texts.append(ex)
                proto_labels.append(intent.intent_name)
        self._proto_labels = proto_labels
        self._proto_emb = self.embedder.encode(proto_texts) if proto_texts else np.zeros((0, 1))

    async def analyze(self, complaint: str, embedding: np.ndarray | None = None) -> QueryMetadata:
        if embedding is None:
            embedding = await asyncio.to_thread(self.embedder.encode, complaint)

        neighbours_task = self.index.nearest(embedding, settings.INTENT_KNN_K)
        sentiment_task = asyncio.to_thread(self._sentiment, complaint)
        neighbours, sentiment = await asyncio.gather(neighbours_task, sentiment_task)

        if self.classifier == "llm" and self.llm is not None and self.llm.available:
            intent = await self._classify_llm(complaint)
            if intent is None:
                intent = self._classify_knn(embedding, neighbours)
        else:
            intent = self._classify_knn(embedding, neighbours)

        same_class = [n for n in neighbours if n.category == intent["intent"]]
        products = self.product_extractor.extract(complaint)
        for inferred in infer_products(same_class, settings.PRODUCT_NEIGHBOUR_MIN_SHARE):
            if inferred not in products:
                products.append(inferred)
        if not products and settings.PRODUCT_LLM_FALLBACK and self.llm is not None and self.llm.available:
            products = await self._extract_products_llm(complaint)

        # Severity base comes from neighbours of the predicted class (all neighbours if none match).
        severity = classify_severity(
            complaint,
            [(n.severity, n.similarity) for n in (same_class or neighbours)],
            sentiment["sentiment_score"],
            intent["is_unknown"],
        )
        return QueryMetadata(
            intent=intent["intent"],
            intent_confidence=round(intent["confidence"], 4),
            products=products,
            severity=severity,
            sentiment=sentiment["sentiment"],
            sentiment_score=round(sentiment["sentiment_score"], 4),
            intent_candidates=intent["candidates"],
            nearest_similarity=(
                round(intent["nearest_similarity"], 4) if intent.get("nearest_similarity") is not None else None
            ),
            is_unknown_intent=intent["is_unknown"],
        )

    def _sentiment(self, complaint: str) -> dict:
        if self.sentiment is None:
            return {"sentiment": "neutral", "sentiment_score": 0.0, "confidence": 0.0}
        return self.sentiment.analyze(complaint)

    def _classify_knn(self, embedding: np.ndarray, neighbours: list[Neighbour]) -> dict:
        labelled = [(n.category, n.similarity) for n in neighbours]
        if len(self._proto_labels):
            proto_sims = self._proto_emb @ embedding
            labelled += list(zip(self._proto_labels, proto_sims.tolist()))
        return knn_vote(labelled, settings.INTENT_MIN_CONFIDENCE, settings.INTENT_MIN_SIMILARITY)

    async def _classify_llm(self, complaint: str) -> dict | None:
        if self.llm is None:
            return None
        cats = [(i.intent_name, i.description) for i in self.intents]
        try:
            result = await self.llm.generate_json(
                [
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": build_intent_user_prompt(complaint, cats)},
                ],
                temperature=0.0,
                max_tokens=200,
            )
        except LLMUnavailable as exc:
            logger.warning("llm intent classification unavailable: %s", exc)
            return None
        label = str(result.get("intent", UNKNOWN_INTENT))
        try:
            confidence = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        valid = {i.intent_name for i in self.intents}
        is_unknown = label not in valid or confidence < settings.INTENT_MIN_CONFIDENCE
        return {
            "intent": UNKNOWN_INTENT if is_unknown else label,
            "confidence": confidence,
            "candidates": [(label, confidence)],
            "nearest_similarity": None,
            "is_unknown": is_unknown,
        }

    async def _extract_products_llm(self, complaint: str) -> list[str]:
        names = [p.product_name for p in self.products]
        if self.llm is None:
            return []
        try:
            result = await self.llm.generate_json(
                [
                    {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Products: {names}\n<complaint>\n{sanitize_for_prompt(complaint)}\n</complaint>",
                    },
                ],
                temperature=0.0,
                max_tokens=100,
            )
        except LLMUnavailable:
            return []
        return [p for p in result.get("products", []) if p in names]
