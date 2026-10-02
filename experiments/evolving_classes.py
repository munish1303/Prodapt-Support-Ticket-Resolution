"""Experiment E4' - Evolving data and ticket classes (use-case requirement 3).

Scenario: a new issue type ("roaming_issue") starts arriving after launch.
  Phase A  send the 60 held-out roaming complaints while the class does not exist.
           -> how are they handled? (intent predicted, decision mix, drift signals)
  Phase B  register the intent (POST /taxonomy/intents) and ingest the wave-2 tickets + KB
           (POST /ingestion) - no retraining, no redeploy.
  Phase C  send the same complaints again.
           -> intent accuracy, roaming KB retrieved, decision mix.
Then the drift report is read. By default wave-2 rows and the intent are removed afterwards
so other evaluations stay reproducible (--keep to leave them).

Runs against the real service container and database (same code paths as the API).
Usage: python experiments/evolving_classes.py [--keep] [--extractive]
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402

from app.api.v1.dependencies import build_container  # noqa: E402
from app.core.database import dispose_engine  # noqa: E402
from app.services import monitoring  # noqa: E402
from app.services.generation import ExtractiveGenerator, GenerationService  # noqa: E402
from app.services.ingestion import IngestionReport, KBArticleIn, TicketIn  # noqa: E402
from app.services.understanding import IntentDef  # noqa: E402
from evaluation.data import PROCESSED_DIR, load_eval, read_jsonl, save_result  # noqa: E402
from experiments.e_common import phase_summary  # noqa: E402

ROAMING = IntentDef(
    "roaming_issue",
    "Problems using mobile service while abroad: roaming data, calls or charges",
    ["data roaming not working abroad", "roaming charges on my bill", "can't call while travelling overseas"],
)


async def run_phase(pipeline, cases, kb_ids):
    rows = []
    for c in cases:
        r = await pipeline.process(c["complaint"], record=True)
        rows.append(
            {
                "intent": r.metadata.intent,
                "decision": r.decision.decision,
                "confidence": r.decision.heuristic_confidence,
                "nearest_similarity": r.metadata.nearest_similarity,
                "max_relevance": max(r.retrieval.relevance_scores) if r.retrieval.items else 0.0,
                "roaming_kb_retrieved": any(d.id in kb_ids for d in r.retrieval.documents),
                "groundedness": r.validation.groundedness_score,
            }
        )
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--extractive", action="store_true", help="force the extractive generator (no LLM calls)")
    args = parser.parse_args()

    container = await build_container()
    ingestion = container.ingestion
    assert ingestion is not None
    if args.extractive:
        container.pipeline.generation = GenerationService(None, ExtractiveGenerator())
    sf = container.session_factory
    cases = load_eval("novel_intent_eval")
    wave2 = [TicketIn(**t) for t in read_jsonl(PROCESSED_DIR / "tickets_wave2.jsonl")]
    kb2 = [KBArticleIn(**a) for a in read_jsonl(PROCESSED_DIR / "kb_articles_wave2.jsonl")]
    kb_ids = {a.article_id for a in kb2}

    async with sf() as s:  # isolate this experiment's monitoring rows
        await s.execute(text("DELETE FROM resolution_requests"))
        await s.commit()
    # Baseline traffic: in-distribution complaints first, so drift has a reference window.
    baseline_cases = load_eval("test")[:60]
    await run_phase(container.pipeline, baseline_cases, kb_ids)
    async with sf() as s:
        await s.execute(text("UPDATE resolution_requests SET created_at = NOW() - interval '3 days'"))
        await s.commit()

    before = await run_phase(container.pipeline, cases, kb_ids)
    async with sf() as s:
        drift_before_ingest = await monitoring.get_drift_report(s, 24)

    report = IngestionReport()
    await ingestion.upsert_intents([ROAMING])
    await ingestion.ingest_tickets(wave2, report)
    await ingestion.ingest_kb_articles(kb2, report)
    await container.refresh_taxonomy()

    after = await run_phase(container.pipeline, cases, kb_ids)

    payload: dict[str, Any] = {
        "n_cases": len(cases),
        "phase_a_before": phase_summary(before, "roaming_issue"),
        "drift_report_after_phase_a": drift_before_ingest,
        "ingestion": report.__dict__,
        "phase_c_after": phase_summary(after, "roaming_issue"),
        "generator": "extractive" if args.extractive else container.pipeline.generation.llm_generator and "llm",
    }
    print(payload["phase_a_before"])
    print(payload["drift_report_after_phase_a"]["alerts"])
    print(payload["phase_c_after"])

    if not args.keep:
        async with sf() as s:
            await s.execute(
                text("DELETE FROM tickets WHERE ticket_id = ANY(:ids)"), {"ids": [t.ticket_id for t in wave2]}
            )
            await s.execute(text("DELETE FROM kb_articles WHERE article_id = ANY(:ids)"), {"ids": list(kb_ids)})
            await s.execute(text("DELETE FROM intent_taxonomy WHERE intent_name = 'roaming_issue'"))
            await s.execute(text("DELETE FROM resolution_requests"))
            await s.commit()
        payload["cleanup"] = "wave-2 rows, roaming intent and experiment request log removed"
    print(f"saved {save_result('evolving_classes', payload)}")
    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
