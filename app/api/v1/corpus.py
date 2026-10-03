"""Read-only views of the knowledge base, so the frontend can show *what RAG retrieves from*.

GET /corpus/stats               table counts, embedding dimension, ANN indexes, category mix
GET /corpus/tickets             paginated, filterable list of historical tickets
GET /corpus/tickets/{id}        one ticket incl. provenance and a preview of its stored embedding
GET /corpus/kb                  KB articles
GET /corpus/kb/{id}             one KB article incl. embedding preview
GET /corpus/search?q=           raw retrieval (no LLM): semantic vs lexical vs hybrid side by side
GET /requests                   recent resolution requests with the sources each one retrieved
"""

from __future__ import annotations

import asyncio
import math

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.dependencies import Container, get_container
from app.config import settings
from app.core.database import get_db
from app.models.schemas import RetrievalResult

router = APIRouter(prefix="/api/v1", tags=["knowledge base"])

PREVIEW_DIMS = 24


def _embedding_preview(vec_text: str | None) -> dict | None:
    """pgvector returns '[0.1,0.2,...]'; expose dimension, L2 norm and the first values (proof it is stored)."""
    if not vec_text:
        return None
    values = [float(x) for x in vec_text.strip("[]").split(",") if x]
    return {
        "dims": len(values),
        "l2_norm": round(math.sqrt(sum(v * v for v in values)), 4),
        "preview": [round(v, 4) for v in values[:PREVIEW_DIMS]],
    }


@router.get("/corpus/stats")
async def corpus_stats(db: AsyncSession = Depends(get_db)) -> dict:
    t = (await db.execute(text("""SELECT count(*), count(embedding), min(created_at), max(created_at),
                          max(vector_dims(embedding)), count(*) FILTER (WHERE (metadata->>'resolved')::boolean IS FALSE)
                   FROM tickets"""))).one()
    k = (await db.execute(text("SELECT count(*), count(embedding) FROM kb_articles"))).one()
    r = (await db.execute(text("SELECT count(*) FROM resolution_requests"))).scalar_one()
    cats = (
        await db.execute(text("SELECT category, count(*) FROM tickets GROUP BY category ORDER BY count(*) DESC"))
    ).all()
    sources = (await db.execute(text("SELECT source, count(*) FROM tickets GROUP BY source ORDER BY 2 DESC"))).all()
    idx = (await db.execute(text("""SELECT tablename, indexname, indexdef FROM pg_indexes
                   WHERE tablename IN ('tickets', 'kb_articles')
                     AND (indexdef ILIKE '%ivfflat%' OR indexdef ILIKE '%hnsw%' OR indexdef ILIKE '%gin%')
                   ORDER BY tablename, indexname"""))).all()
    ext = (await db.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))).scalar()
    pg = (await db.execute(text("SHOW server_version"))).scalar()
    return {
        "tickets": {
            "total": int(t[0]),
            "with_embedding": int(t[1]),
            "unresolved": int(t[5] or 0),
            "date_range": [t[2].isoformat() if t[2] else None, t[3].isoformat() if t[3] else None],
        },
        "kb_articles": {"total": int(k[0]), "with_embedding": int(k[1])},
        "requests_logged": int(r),
        "embedding": {"model": settings.EMBEDDING_MODEL, "dims": int(t[4]) if t[4] else settings.EMBEDDING_DIM},
        "categories": [{"category": c or "(none)", "count": int(n)} for c, n in cats],
        "provenance": [{"source": s or "(unknown)", "count": int(n)} for s, n in sources],
        "indexes": [{"table": tb, "name": nm, "definition": d} for tb, nm, d in idx],
        "database": {"postgres": pg, "pgvector": ext},
        "retrieval_config": {
            "rrf_weights": {"semantic": settings.RRF_SEMANTIC_WEIGHT, "lexical": settings.RRF_LEXICAL_WEIGHT},
            "lexical_query_mode": settings.LEXICAL_QUERY_MODE,
            "top_k": settings.RETRIEVAL_TOP_K,
            "kb_min_slots": settings.RETRIEVAL_KB_MIN_SLOTS,
            "reranking": settings.USE_RERANKING,
        },
    }


@router.get("/corpus/tickets")
async def list_tickets(
    q: str | None = Query(None, max_length=200, description="substring filter on id, complaint or resolution"),
    category: str | None = Query(None, max_length=100),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    where: list[str] = ["1=1"]
    params: dict[str, object] = {"limit": limit, "offset": offset}
    if q:
        where.append("(ticket_id ILIKE :pat OR complaint ILIKE :pat OR resolution ILIKE :pat)")
        params["pat"] = f"%{q}%"
    if category:
        where.append("category = :cat")
        params["cat"] = category
    clause = " AND ".join(where)
    total = (await db.execute(text(f"SELECT count(*) FROM tickets WHERE {clause}"), params)).scalar_one()
    rows = (
        await db.execute(
            text(f"""SELECT ticket_id, complaint, resolution, category, product, severity, sentiment, created_at, source
                    FROM tickets WHERE {clause} ORDER BY ticket_id LIMIT :limit OFFSET :offset"""),
            params,
        )
    ).all()
    return {
        "total": int(total),
        "items": [
            {
                "ticket_id": r[0],
                "complaint": r[1],
                "resolution": r[2],
                "category": r[3],
                "product": r[4],
                "severity": r[5],
                "sentiment": r[6],
                "created_at": r[7].isoformat() if r[7] else None,
                "source": r[8],
            }
            for r in rows
        ],
    }


@router.get("/corpus/tickets/{ticket_id}")
async def get_ticket(ticket_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    r = (
        await db.execute(
            text("""SELECT ticket_id, complaint, resolution, category, product, severity, sentiment, created_at,
                          resolved_at, source, label_source, verified, split, metadata, embedding::text, ingested_at
                   FROM tickets WHERE ticket_id = :id"""),
            {"id": ticket_id},
        )
    ).first()
    if r is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    return {
        "type": "ticket",
        "id": r[0],
        "complaint": r[1],
        "resolution": r[2],
        "category": r[3],
        "product": r[4],
        "severity": r[5],
        "sentiment": r[6],
        "created_at": r[7].isoformat() if r[7] else None,
        "resolved_at": r[8].isoformat() if r[8] else None,
        "provenance": {"source": r[9], "label_source": r[10], "verified": r[11], "split": r[12]},
        "metadata": r[13] or {},
        "embedding": _embedding_preview(r[14]),
        "ingested_at": r[15].isoformat() if r[15] else None,
    }


@router.get("/corpus/kb")
async def list_kb(
    q: str | None = Query(None, max_length=200),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    where = "1=1"
    params: dict[str, object] = {"limit": limit, "offset": offset}
    if q:
        where = "(article_id ILIKE :pat OR title ILIKE :pat OR content ILIKE :pat)"
        params["pat"] = f"%{q}%"
    total = (await db.execute(text(f"SELECT count(*) FROM kb_articles WHERE {where}"), params)).scalar_one()
    rows = (
        await db.execute(
            text(f"""SELECT article_id, title, content, category, product, version, updated_at
                    FROM kb_articles WHERE {where} ORDER BY article_id LIMIT :limit OFFSET :offset"""),
            params,
        )
    ).all()
    return {
        "total": int(total),
        "items": [
            {
                "article_id": r[0],
                "title": r[1],
                "content": r[2],
                "category": r[3],
                "product": r[4],
                "version": r[5],
                "updated_at": r[6].isoformat() if r[6] else None,
            }
            for r in rows
        ],
    }


@router.get("/corpus/kb/{article_id}")
async def get_kb(article_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    r = (
        await db.execute(
            text("""SELECT article_id, title, content, category, product, tags, version, source, metadata,
                          embedding::text, updated_at FROM kb_articles WHERE article_id = :id"""),
            {"id": article_id},
        )
    ).first()
    if r is None:
        raise HTTPException(status_code=404, detail="KB article not found")
    versions = (
        await db.execute(
            text("SELECT count(*) FROM kb_article_versions WHERE article_id = :id"),
            {"id": article_id},
        )
    ).scalar_one()
    return {
        "type": "kb_article",
        "id": r[0],
        "title": r[1],
        "content": r[2],
        "category": r[3],
        "product": r[4],
        "tags": list(r[5] or []),
        "version": r[6],
        "archived_versions": int(versions),
        "provenance": {"source": r[7]},
        "metadata": r[8] or {},
        "embedding": _embedding_preview(r[9]),
        "updated_at": r[10].isoformat() if r[10] else None,
    }


def _hits(result: RetrievalResult) -> list[dict]:
    return [
        {
            "rank": n,
            "id": it.document.id,
            "type": it.document.doc_type,
            "title": it.document.metadata.get("title"),
            "category": it.document.metadata.get("category"),
            "excerpt": " ".join(it.document.text.split())[:200],
            "relevance": round(it.relevance, 4),
            "score": round(it.fused_score, 6),
            "semantic_rank": it.semantic_rank,
            "lexical_rank": it.lexical_rank,
        }
        for n, it in enumerate(result.items, start=1)
    ]


@router.get("/corpus/search")
async def corpus_search(
    q: str = Query(..., min_length=3, max_length=1000),
    k: int = Query(8, ge=1, le=20),
    container: Container = Depends(get_container),
) -> dict:
    """Raw retrieval without the LLM: what each method pulls from the database for this text."""
    retriever = getattr(container.pipeline.retrieval, "retriever", None)
    if retriever is None:
        raise HTTPException(status_code=503, detail="retriever not available")
    embedding = await asyncio.to_thread(container.embedder.encode, q)
    out = {}
    for mode in ("semantic", "lexical", "hybrid"):
        res = await retriever.retrieve(q, top_k=k, mode=mode, query_embedding=embedding, kb_min_slots=0)
        out[mode] = _hits(res)
    return {
        "query": q,
        "k": k,
        "methods": {
            "semantic": f"pgvector cosine similarity (ivfflat) on {settings.EMBEDDING_MODEL.split('/')[-1]} embeddings",
            "lexical": "PostgreSQL full-text search (tsvector, OR-of-terms tsquery, ts_rank)",
            "hybrid": f"weighted reciprocal rank fusion, semantic {settings.RRF_SEMANTIC_WEIGHT} / "
            f"lexical {settings.RRF_LEXICAL_WEIGHT}",
        },
        "results": out,
    }


@router.get("/requests")
async def recent_requests(limit: int = Query(20, ge=1, le=100), db: AsyncSession = Depends(get_db)) -> list[dict]:
    rows = (
        await db.execute(
            text("""SELECT request_id, created_at, complaint_redacted, extracted_metadata, retrieved_sources, decision,
                          heuristic_confidence, groundedness_score, generator, latency_ms
                   FROM resolution_requests ORDER BY created_at DESC LIMIT :limit"""),
            {"limit": limit},
        )
    ).all()
    return [
        {
            "request_id": str(r[0]),
            "created_at": r[1].isoformat() if r[1] else None,
            "complaint": r[2],
            "understanding": r[3] or {},
            "sources": r[4] or [],
            "decision": r[5],
            "confidence": r[6],
            "groundedness": r[7],
            "generator": r[8],
            "latency_ms": r[9],
        }
        for r in rows
    ]
