"""Integration tests against a real PostgreSQL + pgvector (skipped when unreachable).

Uses a throwaway schema-qualified set of rows with a unique id prefix and removes them afterwards,
so it can run against the dev database.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text

from app.config import settings

pytestmark = pytest.mark.db


def _db_available() -> bool:
    async def probe():
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(settings.DATABASE_URL)
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1 FROM tickets LIMIT 1"))
            return True
        except Exception:
            return False
        finally:
            await engine.dispose()

    return asyncio.run(probe())


if not _db_available():
    pytest.skip("database not available", allow_module_level=True)


@pytest.fixture
async def services():
    from app.core.database import dispose_engine, get_session_factory
    from app.models.embeddings import get_embedding_service
    from app.services.ingestion import IngestionService
    from app.services.retrieval import HybridRetriever

    # The corpus holds real MiniLM vectors, so DB tests must embed queries with the same model.
    embedder = get_embedding_service()
    sf = get_session_factory()
    prefix = f"ZZTEST-{uuid.uuid4().hex[:6]}"
    yield sf, IngestionService(sf, embedder), HybridRetriever(sf, embedder), prefix
    async with sf() as s:
        await s.execute(text("DELETE FROM tickets WHERE ticket_id LIKE :p"), {"p": f"{prefix}%"})
        await s.execute(text("DELETE FROM kb_articles WHERE article_id LIKE :p"), {"p": f"{prefix}%"})
        await s.execute(text("DELETE FROM kb_article_versions WHERE article_id LIKE :p"), {"p": f"{prefix}%"})
        await s.commit()
    await dispose_engine()


async def test_ingest_is_idempotent_and_versions_kb(services):
    from app.services.ingestion import KBArticleIn, TicketIn

    sf, ingestion, _, prefix = services
    ticket = TicketIn(
        ticket_id=f"{prefix}-T1",
        complaint="Zebraquux modem flashes purple constantly",
        resolution="Replaced the zebraquux power brick.",
        category="hardware_problem",
    )
    r1 = await ingestion.ingest_tickets([ticket])
    r2 = await ingestion.ingest_tickets([ticket])
    assert r1.tickets_inserted == 1 and r2.tickets_unchanged == 1
    ticket.resolution = "Replaced the zebraquux power brick and updated firmware."
    r3 = await ingestion.ingest_tickets([ticket])
    assert r3.tickets_updated == 1

    art = KBArticleIn(
        article_id=f"{prefix}-KB",
        title="Zebraquux purple light",
        content="If the zebraquux modem flashes purple, replace the power brick. " * 2,
    )
    await ingestion.ingest_kb_articles([art])
    art.content = "Updated: if the zebraquux modem flashes purple, update firmware then replace the brick. " * 2
    r4 = await ingestion.ingest_kb_articles([art])
    assert r4.kb_updated == 1
    async with sf() as s:
        version = (
            await s.execute(text("SELECT version FROM kb_articles WHERE article_id=:i"), {"i": art.article_id})
        ).scalar_one()
        archived = (
            await s.execute(text("SELECT count(*) FROM kb_article_versions WHERE article_id=:i"), {"i": art.article_id})
        ).scalar_one()
    assert version == 2 and archived == 1


async def test_hybrid_finds_new_rows_lexically_and_semantically(services):
    from app.services.ingestion import TicketIn

    _, ingestion, retriever, prefix = services
    await ingestion.ingest_tickets(
        [
            TicketIn(
                ticket_id=f"{prefix}-T2",
                complaint="Zebraquux modem flashes purple constantly",
                resolution="Replaced the zebraquux power brick.",
                category="hardware_problem",
            )
        ]
    )
    for mode in ("lexical", "semantic", "hybrid"):
        result = await retriever.retrieve("zebraquux purple flashing", top_k=10, mode=mode, kb_min_slots=0)
        assert f"{prefix}-T2" in [i.document.id for i in result.items], mode
        assert all(-1.0 <= i.relevance <= 1.0 for i in result.items)


async def test_metadata_filter_restricts_tickets_and_kb(services):
    _, _, retriever, _ = services
    result = await retriever.retrieve(
        "my bill is wrong", top_k=10, mode="hybrid", metadata_filters={"category": "billing_dispute"}, kb_min_slots=2
    )
    assert any(i.document.doc_type == "ticket" for i in result.items)
    assert any(i.document.doc_type == "kb_article" for i in result.items)
    assert all(i.document.metadata["category"] == "billing_dispute" for i in result.items)


async def test_corpus_endpoints_expose_the_knowledge_base(services):
    from types import SimpleNamespace

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from app.api.v1.corpus import router as corpus_router
    from app.api.v1.dependencies import get_container

    sf, _, retriever, _ = services
    app = FastAPI()
    app.include_router(corpus_router)
    app.dependency_overrides[get_container] = lambda: SimpleNamespace(
        embedder=retriever.embedder, pipeline=SimpleNamespace(retrieval=SimpleNamespace(retriever=retriever))
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        stats = (await c.get("/api/v1/corpus/stats")).json()
        assert stats["tickets"]["total"] >= 2000 and stats["embedding"]["dims"] == settings.EMBEDDING_DIM
        assert any("ivfflat" in i["definition"] for i in stats["indexes"])

        page = (await c.get("/api/v1/corpus/tickets", params={"q": "LOS", "limit": 5})).json()
        assert page["total"] > 0 and len(page["items"]) <= 5
        tid = page["items"][0]["ticket_id"]
        ticket = (await c.get(f"/api/v1/corpus/tickets/{tid}")).json()
        assert ticket["embedding"]["dims"] == settings.EMBEDDING_DIM and abs(ticket["embedding"]["l2_norm"] - 1) < 0.01
        assert (await c.get("/api/v1/corpus/tickets/NOPE-1")).status_code == 404

        kb = (await c.get("/api/v1/corpus/kb")).json()
        art = (await c.get(f"/api/v1/corpus/kb/{kb['items'][0]['article_id']}")).json()
        assert art["type"] == "kb_article" and art["embedding"]["dims"] == settings.EMBEDDING_DIM

        res = (await c.get("/api/v1/corpus/search", params={"q": "red LOS light on the fibre box", "k": 5})).json()
        assert set(res["results"]) == {"semantic", "lexical", "hybrid"}
        assert all(len(v) <= 5 for v in res["results"].values()) and res["results"]["hybrid"]

        log = (await c.get("/api/v1/requests", params={"limit": 3})).json()
        assert isinstance(log, list)
