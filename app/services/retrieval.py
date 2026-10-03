"""Retrieval: semantic (pgvector), lexical (PostgreSQL full-text search), hybrid (RRF).

Notes
-----
* Lexical search is PostgreSQL FTS (tsvector/ts_rank), not BM25.
* Long complaints make `plainto_tsquery` (AND of every term) return almost
  nothing, so by default the query is an OR of the complaint's content terms
  and ranking does the work (LEXICAL_QUERY_MODE=or|and; compared in E1).
* Fused RRF scores are only for ordering. Evidence signals downstream use
  `relevance` = cosine similarity between the query and each returned document,
  computed for every returned doc (including lexical-only hits).
"""

from __future__ import annotations

import asyncio
import re
from typing import Protocol
from collections.abc import Sequence

import numpy as np
from sqlalchemy import text

from app.config import settings
from app.models.embeddings import EmbeddingService, to_pgvector
from app.models.schemas import Document, RetrievalResult, RetrievedDocument
from app.utils.text import STOPWORDS

TABLES = {
    "tickets": {"id": "ticket_id", "type": "ticket"},
    "kb_articles": {"id": "article_id", "type": "kb_article"},
}
_TSQUERY_TOKEN = re.compile(r"[a-z0-9]+")
ALLOWED_FILTERS = {"product", "category"}


def build_or_tsquery(query_text: str, max_terms: int = 40) -> str:
    seen, terms = set(), []
    for tok in _TSQUERY_TOKEN.findall(query_text.lower()):
        if tok in STOPWORDS or len(tok) < 2 or tok in seen:
            continue
        seen.add(tok)
        terms.append(tok)
    return " | ".join(terms[:max_terms])


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """score(d) = Σ_i w_i / (k + rank_i(d)), ranks starting at 1."""
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for weight, ranked in zip(weights, ranked_lists):
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def _filter_clause(filters: dict | None, params: dict) -> str:
    clause = ""
    for key, value in (filters or {}).items():
        if key in ALLOWED_FILTERS and value:
            clause += f" AND {key} = :f_{key}"
            params[f"f_{key}"] = value
    return clause


class SemanticSearcher:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def search(self, table: str, query_vec: str, k: int, filters: dict | None = None) -> list[tuple[str, float]]:
        id_col = TABLES[table]["id"]
        params = {"v": query_vec, "k": k}
        where = _filter_clause(filters, params)
        sql = f"""
            SELECT {id_col}, 1 - (embedding <=> CAST(:v AS vector)) AS sim
            FROM {table} WHERE embedding IS NOT NULL {where}
            ORDER BY embedding <=> CAST(:v AS vector) LIMIT :k
        """
        async with self.session_factory() as session:
            await session.execute(text(f"SET LOCAL ivfflat.probes = {int(settings.IVFFLAT_PROBES)}"))
            rows = (await session.execute(text(sql), params)).all()
        return [(r[0], float(r[1])) for r in rows]


class LexicalSearcher:
    def __init__(self, session_factory, mode: str | None = None):
        self.session_factory = session_factory
        self.mode = mode or settings.LEXICAL_QUERY_MODE

    async def search(self, table: str, query_text: str, k: int, filters: dict | None = None) -> list[tuple[str, float]]:
        id_col = TABLES[table]["id"]
        params: dict = {"k": k}
        if self.mode == "and":
            tsq = "plainto_tsquery('english', :q)"
            params["q"] = query_text
        else:
            or_query = build_or_tsquery(query_text)
            if not or_query:
                return []
            tsq = "to_tsquery('english', :q)"
            params["q"] = or_query
        where = _filter_clause(filters, params)
        sql = f"""
            SELECT {id_col}, ts_rank(text_search, {tsq}, 1) AS rank
            FROM {table} WHERE text_search @@ {tsq} {where}
            ORDER BY rank DESC LIMIT :k
        """
        async with self.session_factory() as session:
            rows = (await session.execute(text(sql), params)).all()
        return [(r[0], float(r[1])) for r in rows]


class HybridRetriever:
    """Runs semantic and/or lexical search per table, fuses, merges tables, fetches documents."""

    def __init__(
        self,
        session_factory,
        embedder: EmbeddingService,
        lexical_mode: str | None = None,
        rrf_k: int | None = None,
        semantic_weight: float | None = None,
        lexical_weight: float | None = None,
    ):
        self.session_factory = session_factory
        self.embedder = embedder
        self.semantic = SemanticSearcher(session_factory)
        self.lexical = LexicalSearcher(session_factory, lexical_mode)
        self.rrf_k = rrf_k or settings.RRF_K
        self.semantic_weight = settings.RRF_SEMANTIC_WEIGHT if semantic_weight is None else semantic_weight
        self.lexical_weight = settings.RRF_LEXICAL_WEIGHT if lexical_weight is None else lexical_weight

    async def retrieve(
        self,
        query_text: str,
        top_k: int | None = None,
        mode: str = "hybrid",
        query_embedding: np.ndarray | None = None,
        metadata_filters: dict | None = None,
        kb_min_slots: int | None = None,
    ) -> RetrievalResult:
        top_k = top_k or settings.RETRIEVAL_TOP_K
        kb_min_slots = settings.RETRIEVAL_KB_MIN_SLOTS if kb_min_slots is None else kb_min_slots
        if query_embedding is None:
            query_embedding = await asyncio.to_thread(self.embedder.encode, query_text)
        qvec = to_pgvector(query_embedding)
        # Only the database work is timed (not the embedding above): a hung database fails fast with TimeoutError.
        return await asyncio.wait_for(
            self._search(query_text, qvec, top_k, mode, metadata_filters, kb_min_slots),
            timeout=settings.DB_QUERY_TIMEOUT_S,
        )

    async def _search(
        self,
        query_text: str,
        qvec: str,
        top_k: int,
        mode: str,
        metadata_filters: dict | None,
        kb_min_slots: int,
    ) -> RetrievalResult:
        limits = {"tickets": settings.RETRIEVAL_CANDIDATES_TICKETS, "kb_articles": settings.RETRIEVAL_CANDIDATES_KB}
        tasks = {}
        for table, limit in limits.items():
            if mode in ("semantic", "hybrid"):
                tasks[(table, "sem")] = self.semantic.search(table, qvec, limit, metadata_filters)
            if mode in ("lexical", "hybrid"):
                tasks[(table, "lex")] = self.lexical.search(table, query_text, limit, metadata_filters)
        results = dict(zip(tasks.keys(), await asyncio.gather(*tasks.values())))

        # Build ONE semantic ranking and ONE lexical ranking across both tables, then fuse once.
        # (Fusing per table and interleaving is wrong: RRF depends only on rank, so the #1 KB article
        # out of 40 would tie the #1 ticket out of 2,000 and KB articles would flood the top-k. E1 v1.)
        table_of: dict[str, str] = {}
        sem_all: list[tuple[str, float]] = []
        lex_all: list[tuple[str, float]] = []
        for table in limits:
            for doc_id, score in results.get((table, "sem"), []):
                sem_all.append((doc_id, score))
                table_of[doc_id] = table
            for doc_id, score in results.get((table, "lex"), []):
                lex_all.append((doc_id, score))
                table_of[doc_id] = table
        sem_all.sort(key=lambda x: -x[1])
        lex_all.sort(key=lambda x: -x[1])
        sem_rank = {d: i for i, (d, _) in enumerate(sem_all, 1)}
        lex_rank = {d: i for i, (d, _) in enumerate(lex_all, 1)}
        if mode == "semantic":
            scored = sem_all
        elif mode == "lexical":
            scored = lex_all
        else:
            scored = reciprocal_rank_fusion(
                [[d for d, _ in sem_all], [d for d, _ in lex_all]],
                [self.semantic_weight, self.lexical_weight],
                self.rrf_k,
            )
        ranked = [(table_of[d], d, s, sem_rank.get(d), lex_rank.get(d)) for d, s in scored]
        merged = self._apply_kb_slots(ranked, top_k, kb_min_slots)
        items = await self._fetch(merged, qvec)
        return RetrievalResult(items=items, method=f"{mode}" + ("" if mode != "lexical" else f"_{self.lexical.mode}"))

    @staticmethod
    def _apply_kb_slots(ranked: list[tuple], top_k: int, kb_min_slots: int) -> list[tuple]:
        """Take the top_k of an already-ranked list, guaranteeing at least kb_min_slots KB articles
        (agents want the canonical procedure even when similar tickets outrank it)."""
        chosen = ranked[:top_k]
        kb_chosen = [r for r in chosen if r[0] == "kb_articles"]
        if len(kb_chosen) < kb_min_slots:
            extra_kb = [r for r in ranked[top_k:] if r[0] == "kb_articles"][: kb_min_slots - len(kb_chosen)]
            if extra_kb:
                tickets_in = [r for r in chosen if r[0] == "tickets"]
                keep = tickets_in[: max(0, len(tickets_in) - len(extra_kb))]
                chosen = [r for r in chosen if r in keep or r in kb_chosen] + extra_kb
        return chosen

    async def _fetch(self, merged, qvec: str) -> list[RetrievedDocument]:
        ticket_ids = [r[1] for r in merged if r[0] == "tickets"]
        kb_ids = [r[1] for r in merged if r[0] == "kb_articles"]
        docs: dict[str, tuple[Document, float]] = {}
        async with self.session_factory() as session:
            if ticket_ids:
                rows = (
                    await session.execute(
                        text(
                            """SELECT ticket_id, complaint, resolution, category, product, severity, created_at, metadata,
                              1 - (embedding <=> CAST(:v AS vector))
                       FROM tickets WHERE ticket_id = ANY(:ids)"""
                        ),
                        {"v": qvec, "ids": ticket_ids},
                    )
                ).all()
                for r in rows:
                    docs[r[0]] = (
                        Document(
                            id=r[0],
                            text=f"Complaint: {r[1]}\nResolution: {r[2] or 'No resolution recorded.'}",
                            metadata={
                                "type": "ticket",
                                "category": r[3],
                                "product": r[4],
                                "severity": r[5],
                                "created_at": r[6].isoformat() if r[6] else None,
                                "resolved": bool((r[7] or {}).get("resolved", r[2] is not None)),
                                "scenario_id": (r[7] or {}).get("scenario_id"),
                            },
                        ),
                        float(r[8]),
                    )
            if kb_ids:
                rows = (
                    await session.execute(
                        text("""SELECT article_id, title, content, category, product, version, metadata,
                              1 - (embedding <=> CAST(:v AS vector))
                       FROM kb_articles WHERE article_id = ANY(:ids)"""),
                        {"v": qvec, "ids": kb_ids},
                    )
                ).all()
                for r in rows:
                    docs[r[0]] = (
                        Document(
                            id=r[0],
                            text=f"{r[1]}\n{r[2]}",
                            metadata={
                                "type": "kb_article",
                                "title": r[1],
                                "category": r[3],
                                "product": r[4],
                                "version": r[5],
                                "scenario_id": (r[6] or {}).get("scenario_id"),
                            },
                        ),
                        float(r[7]),
                    )
        items = []
        for table, doc_id, score, s_rank, l_rank in merged:
            if doc_id in docs:
                doc, rel = docs[doc_id]
                items.append(RetrievedDocument(doc, float(score), rel, s_rank, l_rank))
        return items


class Reranker:
    """Cross-encoder reranker (enabled only if Experiment 2 justifies it)."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name or settings.RERANKER_MODEL, device="cpu")

    def rerank(self, query: str, items: list[RetrievedDocument], top_k: int) -> list[RetrievedDocument]:
        if not items:
            return []
        scores = self.model.predict([[query, i.document.text[:2000]] for i in items], show_progress_bar=False)
        for item, score in zip(items, scores):
            item.fused_score = float(score)
        return sorted(items, key=lambda i: -i.fused_score)[:top_k]


class SupportsRetrieve(Protocol):
    async def retrieve(
        self,
        query_text: str,
        top_k: int | None = None,
        query_embedding: np.ndarray | None = None,
        metadata_filters: dict | None = None,
    ) -> RetrievalResult: ...


class RetrievalService:
    def __init__(self, retriever: HybridRetriever, reranker: Reranker | None = None, use_reranking: bool | None = None):
        self.retriever = retriever
        self.reranker = reranker
        self.use_reranking = settings.USE_RERANKING if use_reranking is None else use_reranking

    async def retrieve(
        self,
        query_text: str,
        top_k: int | None = None,
        query_embedding: np.ndarray | None = None,
        metadata_filters: dict | None = None,
    ) -> RetrievalResult:
        top_k = top_k or settings.RETRIEVAL_TOP_K
        reranker = self.reranker if self.use_reranking else None
        k = settings.RERANK_CANDIDATES if reranker is not None else top_k
        result = await self.retriever.retrieve(query_text, k, "hybrid", query_embedding, metadata_filters)
        if reranker is not None:
            result.items = await asyncio.to_thread(reranker.rerank, query_text, result.items, top_k)
            result.method = "hybrid_reranked"
        return result
