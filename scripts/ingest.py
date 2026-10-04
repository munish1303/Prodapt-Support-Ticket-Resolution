"""Bulk-ingest tickets and KB articles from JSONL files, seed taxonomies, build ANN indexes.

Usage:
  python scripts/ingest.py                         # wave-1 corpus (default files)
  python scripts/ingest.py --tickets data/processed/tickets_wave2.jsonl \
                           --kb data/processed/kb_articles_wave2.jsonl --no-taxonomy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.database import dispose_engine, get_session_factory  # noqa: E402
from app.models.embeddings import get_embedding_service  # noqa: E402
from app.services.ingestion import IngestionReport, IngestionService, KBArticleIn, TicketIn  # noqa: E402
from app.services.understanding import load_taxonomy_files  # noqa: E402
from scripts.init_db import build_indexes  # noqa: E402
from scripts.reembed import ensure_vector_schema  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickets", default=str(ROOT / "data/processed/tickets.jsonl"))
    parser.add_argument("--kb", default=str(ROOT / "data/processed/kb_articles.jsonl"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-taxonomy", action="store_true")
    parser.add_argument("--no-indexes", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    service = IngestionService(get_session_factory(), get_embedding_service(), args.batch_size)
    try:
        await ensure_vector_schema()  # vector columns sized for the configured embedding model
        if not args.no_taxonomy:
            intents, products = load_taxonomy_files()
            await service.upsert_intents(intents)
            await service.upsert_products(products)
            print(f"taxonomy: {len(intents)} intents, {len(products)} products")
        report = IngestionReport()
        tickets = [TicketIn(**t) for t in read_jsonl(Path(args.tickets))] if args.tickets else []
        kb = [KBArticleIn(**a) for a in read_jsonl(Path(args.kb))] if args.kb else []
        await service.ingest_tickets(tickets, report)
        await service.ingest_kb_articles(kb, report)
        print(json.dumps(report.__dict__, indent=2))
        if not args.no_indexes:
            await build_indexes()
    finally:
        await dispose_engine()
    print(f"done in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
