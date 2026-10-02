"""Create the schema (idempotent) and, optionally, (re)build ANN indexes sized from row counts.

Usage:
  python scripts/init_db.py                  # apply migrations
  python scripts/init_db.py --build-indexes  # (re)create ivfflat indexes, lists = max(10, sqrt(rows))
  python scripts/init_db.py --reset          # DROP all tables first (destructive, local dev only)
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402

from app.core.database import dispose_engine, get_engine  # noqa: E402

TABLES = ["resolution_requests", "kb_article_versions", "kb_articles", "tickets", "intent_taxonomy", "product_taxonomy"]
ANN_INDEXES = [
    ("tickets_embedding_idx", "tickets", "embedding"),
    ("tickets_complaint_embedding_idx", "tickets", "complaint_embedding"),
    ("kb_embedding_idx", "kb_articles", "embedding"),
]


async def apply_migrations(reset: bool = False) -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        if reset:
            for t in TABLES:
                await conn.execute(text(f"DROP TABLE IF EXISTS {t} CASCADE"))
        for path in sorted((ROOT / "migrations").glob("*.sql")):
            sql = path.read_text(encoding="utf-8")
            # asyncpg cannot run multiple statements with parameters; split on ';' at line ends.
            for stmt in [s.strip() for s in sql.split(";\n") if s.strip()]:
                lines = [ln for ln in stmt.splitlines() if not ln.strip().startswith("--")]
                if "".join(lines).strip():
                    await conn.execute(text("\n".join(lines)))
            print(f"applied {path.name}")


async def build_indexes() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        for name, table, column in ANN_INDEXES:
            n = (await conn.execute(text(f"SELECT count(*) FROM {table} WHERE {column} IS NOT NULL"))).scalar_one()
            lists = max(10, int(math.sqrt(max(n, 1))))
            await conn.execute(text(f"DROP INDEX IF EXISTS {name}"))
            await conn.execute(
                text(
                    f"CREATE INDEX {name} ON {table} USING ivfflat ({column} vector_cosine_ops) WITH (lists = {lists})"
                )
            )
            print(f"{name}: rows={n} lists={lists}")
        await conn.execute(text("ANALYZE tickets"))
        await conn.execute(text("ANALYZE kb_articles"))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--build-indexes", action="store_true")
    args = parser.parse_args()
    try:
        if args.build_indexes:
            await build_indexes()
        else:
            await apply_migrations(args.reset)
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
