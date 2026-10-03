"""Keep stored vectors consistent with the configured embedding model.

The stored vectors are only meaningful for the model that produced them. This script compares
system_meta.embedding_model / the vector column sizes with settings.EMBEDDING_MODEL / EMBEDDING_DIM and, if they
differ:
  1. resizes the vector columns when the dimension changed (the old vectors are dropped with them),
     or clears them when only the model changed;
  2. re-embeds every row whose vector is missing, in small committed batches (safe to interrupt and re-run: it
     resumes where it stopped, which matters on a memory-constrained machine);
  3. rebuilds the ivfflat indexes (lists = sqrt(rows));
  4. records the model in system_meta.
When everything already matches it does nothing and loads no model, so it runs on every container start.

Usage:
  python scripts/reembed.py             # only if needed
  python scripts/reembed.py --dry-run   # print the plan
  python scripts/reembed.py --force     # re-embed everything with the configured model
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.database import dispose_engine, get_engine  # noqa: E402
from app.models.embeddings import to_pgvector  # noqa: E402
from app.services.ingestion import kb_text, ticket_text  # noqa: E402
from scripts.init_db import build_indexes  # noqa: E402

VECTOR_COLUMNS = [("tickets", "embedding"), ("tickets", "complaint_embedding"), ("kb_articles", "embedding")]


@dataclass
class Plan:
    resize: list[tuple[str, str]]  # columns whose dimension differs from the configured one
    clear: bool  # same dimension but a different model: existing vectors must be replaced
    record_only: bool  # vectors already match; only system_meta needs writing

    @property
    def changes_vectors(self) -> bool:
        return bool(self.resize) or self.clear


def plan_change(
    column_dims: dict[tuple[str, str], int | None],
    stored_model: str | None,
    target_model: str,
    target_dim: int,
) -> Plan:
    """Pure decision logic (unit-tested).

    stored_model is None for databases built before system_meta existed; their vectors are assumed to come from the
    configured model when the dimension matches (the only model used then was the 384-d default, so a dimension
    mismatch is what identifies a model change for them).
    """
    resize = [col for col, dim in column_dims.items() if dim != target_dim]
    clear = not resize and stored_model is not None and stored_model != target_model
    record_only = not resize and not clear and stored_model != target_model
    return Plan(resize, clear, record_only)


async def current_state(conn) -> tuple[dict[tuple[str, str], int | None], str | None]:
    dims: dict[tuple[str, str], int | None] = {}
    for table, column in VECTOR_COLUMNS:
        # pgvector stores the declared dimension in atttypmod
        dims[(table, column)] = (
            await conn.execute(
                text("SELECT atttypmod FROM pg_attribute WHERE attrelid = to_regclass(:t) AND attname = :c"),
                {"t": table, "c": column},
            )
        ).scalar()
    model = (await conn.execute(text("SELECT value FROM system_meta WHERE key = 'embedding_model'"))).scalar()
    return dims, model


async def ensure_vector_schema() -> Plan:
    """Apply the cheap part of the plan (resize/clear + metadata). Returns the plan so callers can re-embed."""
    async with get_engine().begin() as conn:
        dims, stored = await current_state(conn)
        plan = plan_change(dims, stored, settings.EMBEDDING_MODEL, settings.EMBEDDING_DIM)
        for table, column in plan.resize:
            # Dropping the column also drops its ivfflat index; the old vectors cannot be cast to a new size.
            await conn.execute(text(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {column}"))
            await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} vector({settings.EMBEDDING_DIM})"))
            print(f"resized {table}.{column}: {dims[(table, column)]} -> {settings.EMBEDDING_DIM} dims")
        if plan.clear:
            for table, column in VECTOR_COLUMNS:
                await conn.execute(text(f"UPDATE {table} SET {column} = NULL"))
            print(f"cleared vectors produced by {stored}")
        if plan.changes_vectors:
            # Mark the migration as in progress: rows are re-embedded with the new model from here on.
            await _set_meta(conn, "embedding_model", settings.EMBEDDING_MODEL)
            await _set_meta(conn, "embedding_dim", str(settings.EMBEDDING_DIM))
            await _set_meta(conn, "reembed_complete", "false")
        elif plan.record_only:
            await _set_meta(conn, "embedding_model", settings.EMBEDDING_MODEL)
            await _set_meta(conn, "embedding_dim", str(settings.EMBEDDING_DIM))
    return plan


async def _set_meta(conn, key: str, value: str) -> None:
    await conn.execute(
        text("""INSERT INTO system_meta (key, value) VALUES (:k, :v)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()"""),
        {"k": key, "v": value},
    )


async def missing_counts(conn) -> tuple[int, int]:
    t = (
        await conn.execute(text("SELECT count(*) FROM tickets WHERE embedding IS NULL OR complaint_embedding IS NULL"))
    ).scalar_one()
    k = (await conn.execute(text("SELECT count(*) FROM kb_articles WHERE embedding IS NULL"))).scalar_one()
    return int(t), int(k)


async def reembed_missing(batch_size: int) -> int:
    from app.models.embeddings import get_embedding_service

    embedder = get_embedding_service()
    done, started = 0, time.perf_counter()
    while True:  # tickets: complaint+resolution vector and complaint-only vector
        async with get_engine().begin() as conn:
            rows = (
                await conn.execute(
                    text("""SELECT ticket_id, complaint, resolution FROM tickets
                            WHERE embedding IS NULL OR complaint_embedding IS NULL ORDER BY ticket_id LIMIT :n"""),
                    {"n": batch_size},
                )
            ).all()
            if not rows:
                break
            full = await asyncio.to_thread(embedder.encode, [ticket_text(r[1], r[2]) for r in rows])
            comp = await asyncio.to_thread(embedder.encode, [r[1] for r in rows])
            for r, f, c in zip(rows, full, comp):
                await conn.execute(
                    text("""UPDATE tickets SET embedding = CAST(:f AS vector), complaint_embedding = CAST(:c AS vector)
                            WHERE ticket_id = :id"""),
                    {"f": to_pgvector(f), "c": to_pgvector(c), "id": r[0]},
                )
        done += len(rows)
        print(f"tickets re-embedded: {done} ({time.perf_counter() - started:.0f}s)", flush=True)
    async with get_engine().begin() as conn:
        rows = (
            await conn.execute(text("SELECT article_id, title, content FROM kb_articles WHERE embedding IS NULL"))
        ).all()
        if rows:
            vecs = await asyncio.to_thread(embedder.encode, [kb_text(r[1], r[2]) for r in rows])
            for r, v in zip(rows, vecs):
                await conn.execute(
                    text("UPDATE kb_articles SET embedding = CAST(:v AS vector) WHERE article_id = :id"),
                    {"v": to_pgvector(v), "id": r[0]},
                )
            print(f"KB articles re-embedded: {len(rows)}", flush=True)
    return done + len(rows)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    try:
        if args.dry_run:
            async with get_engine().connect() as conn:
                dims, stored = await current_state(conn)
                print(plan_change(dims, stored, settings.EMBEDDING_MODEL, settings.EMBEDDING_DIM))
                print("missing (tickets, kb):", await missing_counts(conn))
            return
        if args.force:
            async with get_engine().begin() as conn:
                for table, column in VECTOR_COLUMNS:
                    await conn.execute(text(f"UPDATE {table} SET {column} = NULL"))
                await _set_meta(conn, "reembed_complete", "false")
        plan = await ensure_vector_schema()
        async with get_engine().connect() as conn:
            missing = await missing_counts(conn)
            complete = (
                await conn.execute(text("SELECT value FROM system_meta WHERE key = 'reembed_complete'"))
            ).scalar()
        if sum(missing) == 0 and complete != "false":
            print(f"embeddings up to date ({settings.EMBEDDING_MODEL})")
            return
        if sum(missing):
            print(f"re-embedding with {settings.EMBEDDING_MODEL}: {missing[0]} tickets, {missing[1]} KB articles")
            await reembed_missing(args.batch_size)
        await build_indexes()
        async with get_engine().begin() as conn:
            await _set_meta(conn, "reembed_complete", "true")
        print(f"done: vectors now from {settings.EMBEDDING_MODEL} (plan: {plan})")
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
