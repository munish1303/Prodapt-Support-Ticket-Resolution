"""Ingestion of tickets, KB articles and taxonomy entries (plan §13).

Incremental by design: tickets are upserted by id (a changed resolution
re-embeds the row), KB articles are versioned (the previous version is archived
in kb_article_versions), and new intent classes can be registered at runtime.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text

from app.models.embeddings import EmbeddingService, to_pgvector
from app.services.understanding import IntentDef, ProductDef

logger = logging.getLogger(__name__)

SEVERITIES = {"critical", "high", "medium", "low"}


class TicketIn(BaseModel):
    ticket_id: str = Field(..., min_length=1, max_length=50)
    complaint: str = Field(..., min_length=10, max_length=5000)
    resolution: str | None = Field(None, max_length=10000)
    category: str | None = Field(None, max_length=100)
    product: str | None = Field(None, max_length=100)
    severity: str | None = None
    sentiment: str | None = None
    sentiment_score: float | None = None
    created_at: datetime | None = None
    resolved_at: datetime | None = None
    source: str = "unknown"
    label_source: str = "unknown"
    verified: bool = False
    split: str = "retrieval_corpus"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("complaint")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 10:
            raise ValueError("complaint must be at least 10 characters")
        return v

    @field_validator("severity")
    @classmethod
    def _severity(cls, v: str | None) -> str | None:
        if v is not None and v not in SEVERITIES:
            raise ValueError(f"invalid severity: {v}")
        return v


class KBArticleIn(BaseModel):
    article_id: str = Field(..., min_length=1, max_length=50)
    title: str = Field(..., min_length=3, max_length=500)
    content: str = Field(..., min_length=50, max_length=50000)
    category: str | None = None
    product: str | None = None
    tags: list[str] = Field(default_factory=list)
    source: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class IngestionReport:
    tickets_inserted: int = 0
    tickets_updated: int = 0
    tickets_unchanged: int = 0
    kb_inserted: int = 0
    kb_updated: int = 0
    kb_unchanged: int = 0
    embeddings_generated: int = 0
    new_categories: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def ticket_embedding_text(t: TicketIn) -> str:
    return f"{t.complaint} {t.resolution or ''}".strip()


class IngestionService:
    def __init__(self, session_factory, embedder: EmbeddingService, batch_size: int = 64):
        self.session_factory = session_factory
        self.embedder = embedder
        self.batch_size = batch_size

    async def ingest_tickets(self, tickets: list[TicketIn], report: IngestionReport | None = None) -> IngestionReport:
        report = report or IngestionReport()
        if not tickets:
            return report
        async with self.session_factory() as session:
            ids = [t.ticket_id for t in tickets]
            rows = (
                await session.execute(
                    text(
                        "SELECT ticket_id, complaint, coalesce(resolution,'') FROM tickets WHERE ticket_id = ANY(:ids)"
                    ),
                    {"ids": ids},
                )
            ).all()
            existing = {r[0]: (r[1], r[2]) for r in rows}
            known_cats = {r[0] for r in (await session.execute(text("SELECT DISTINCT category FROM tickets"))).all()}

        to_write = []
        for t in tickets:
            prev = existing.get(t.ticket_id)
            if prev is not None and prev == (t.complaint, t.resolution or ""):
                report.tickets_unchanged += 1
                continue
            to_write.append((t, prev is not None))
        new_cats = sorted({t.category for t, _ in to_write if t.category and t.category not in known_cats})
        report.new_categories.extend(c for c in new_cats if c not in report.new_categories)

        for start in range(0, len(to_write), self.batch_size):
            batch = to_write[start : start + self.batch_size]
            full_vecs, complaint_vecs = await asyncio.gather(
                asyncio.to_thread(self.embedder.encode, [ticket_embedding_text(t) for t, _ in batch]),
                asyncio.to_thread(self.embedder.encode, [t.complaint for t, _ in batch]),
            )
            report.embeddings_generated += 2 * len(batch)
            async with self.session_factory() as session:
                for (t, is_update), fv, cv in zip(batch, full_vecs, complaint_vecs):
                    try:
                        async with session.begin_nested():  # savepoint: one bad row doesn't abort the batch
                            await session.execute(
                                text("""
                            INSERT INTO tickets (ticket_id, complaint, resolution, category, product, severity, sentiment,
                                sentiment_score, created_at, resolved_at, source, label_source, verified, split, metadata,
                                embedding, complaint_embedding)
                            VALUES (:ticket_id, :complaint, :resolution, :category, :product, :severity, :sentiment,
                                :sentiment_score, :created_at, :resolved_at, :source, :label_source, :verified, :split,
                                CAST(:metadata AS jsonb), CAST(:emb AS vector), CAST(:cemb AS vector))
                            ON CONFLICT (ticket_id) DO UPDATE SET
                                complaint = EXCLUDED.complaint, resolution = EXCLUDED.resolution,
                                category = EXCLUDED.category, product = EXCLUDED.product, severity = EXCLUDED.severity,
                                sentiment = EXCLUDED.sentiment, resolved_at = EXCLUDED.resolved_at,
                                metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding,
                                complaint_embedding = EXCLUDED.complaint_embedding, ingested_at = NOW()
                            """),
                                {
                                    **t.model_dump(exclude={"metadata"}),
                                    "metadata": json.dumps(t.metadata),
                                    "emb": to_pgvector(fv),
                                    "cemb": to_pgvector(cv),
                                },
                            )
                        if is_update:
                            report.tickets_updated += 1
                        else:
                            report.tickets_inserted += 1
                    except Exception as exc:  # keep the batch going; report the row
                        report.errors.append(f"{t.ticket_id}: {exc.__class__.__name__}: {str(exc)[:200]}")
                await session.commit()
        return report

    async def ingest_kb_articles(
        self, articles: list[KBArticleIn], report: IngestionReport | None = None
    ) -> IngestionReport:
        report = report or IngestionReport()
        if not articles:
            return report
        async with self.session_factory() as session:
            rows = (
                await session.execute(
                    text("SELECT article_id, title, content, version FROM kb_articles WHERE article_id = ANY(:ids)"),
                    {"ids": [a.article_id for a in articles]},
                )
            ).all()
        existing = {r[0]: (r[1], r[2], r[3]) for r in rows}
        changed = [a for a in articles if existing.get(a.article_id, (None, None))[:2] != (a.title, a.content)]
        report.kb_unchanged += len(articles) - len(changed)
        if not changed:
            return report
        vecs = await asyncio.to_thread(self.embedder.encode, [f"{a.title}\n{a.content}" for a in changed])
        report.embeddings_generated += len(changed)
        async with self.session_factory() as session:
            for a, v in zip(changed, vecs):
                prev = existing.get(a.article_id)
                try:
                    async with session.begin_nested():
                        if prev:
                            await session.execute(
                                text(
                                    "INSERT INTO kb_article_versions (article_id, version, title, content) VALUES (:id, :ver, :t, :c) "
                                    "ON CONFLICT DO NOTHING"
                                ),
                                {"id": a.article_id, "ver": prev[2], "t": prev[0], "c": prev[1]},
                            )
                        await session.execute(
                            text("""
                        INSERT INTO kb_articles (article_id, title, content, category, product, tags, version, source, metadata, embedding)
                        VALUES (:article_id, :title, :content, :category, :product, :tags, 1, :source, CAST(:metadata AS jsonb), CAST(:emb AS vector))
                        ON CONFLICT (article_id) DO UPDATE SET
                            title = EXCLUDED.title, content = EXCLUDED.content, category = EXCLUDED.category,
                            product = EXCLUDED.product, tags = EXCLUDED.tags, metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding, version = kb_articles.version + 1, updated_at = NOW()
                        """),
                            {
                                **a.model_dump(exclude={"metadata"}),
                                "metadata": json.dumps(a.metadata),
                                "emb": to_pgvector(v),
                            },
                        )
                        if prev:
                            report.kb_updated += 1
                        else:
                            report.kb_inserted += 1
                except Exception as exc:
                    report.errors.append(f"{a.article_id}: {exc.__class__.__name__}: {str(exc)[:200]}")
            await session.commit()
        return report

    async def upsert_intents(self, intents: list[IntentDef]) -> None:
        async with self.session_factory() as session:
            for i in intents:
                await session.execute(
                    text("""INSERT INTO intent_taxonomy (intent_name, description, examples)
                       VALUES (:n, :d, CAST(:e AS jsonb))
                       ON CONFLICT (intent_name) DO UPDATE SET description = EXCLUDED.description,
                           examples = EXCLUDED.examples, active = TRUE"""),
                    {"n": i.intent_name, "d": i.description, "e": json.dumps(i.examples)},
                )
            await session.commit()

    async def upsert_products(self, products: list[ProductDef]) -> None:
        async with self.session_factory() as session:
            for p in products:
                await session.execute(
                    text(
                        """INSERT INTO product_taxonomy (product_name, category, aliases) VALUES (:n, :c, :a)
                       ON CONFLICT (product_name) DO UPDATE SET category = EXCLUDED.category, aliases = EXCLUDED.aliases, active = TRUE"""
                    ),
                    {"n": p.product_name, "c": p.category, "a": p.aliases},
                )
            await session.commit()
