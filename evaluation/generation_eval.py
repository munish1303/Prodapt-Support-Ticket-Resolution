"""End-to-end generation evaluation (pipeline level).

For each held-out complaint (data/evaluation/generation_eval.jsonl) the full
pipeline runs (understanding -> hybrid retrieval -> generation -> validation -> decision).

Metrics
  citation_accuracy / citation_coverage / groundedness   (from the validation service)
  reference_step_recall   fraction of the scenario's reference fix steps matched by a
                          generated step (MiniLM cosine >= --match-threshold)
  step_precision          fraction of generated steps that match some reference step
  kb_retrieved            the scenario's KB article is among the sources
  decision distribution, mean heuristic confidence, latency
  confidence_quality_auc  AUROC of heuristic confidence for predicting a "good" draft
                          (reference_step_recall >= 0.5) - does the score mean anything?
  novel_intent decisions  wave-2 complaints (class absent from corpus): how often does
                          the pipeline RESOLVE something it has never seen?
  optional --judge        LLM-as-judge scores (directional only; known biases)

Usage:
  python evaluation/generation_eval.py --generator extractive
  python evaluation/generation_eval.py --generator llm [--judge] [--limit 50]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score  # noqa: E402

from app.api.v1.dependencies import build_container  # noqa: E402
from app.config import settings  # noqa: E402
from app.core.database import dispose_engine  # noqa: E402
from app.core.llm import LLMUnavailable, OpenAICompatibleProvider  # noqa: E402
from app.core.prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt  # noqa: E402
from app.models.embeddings import get_embedding_service  # noqa: E402
from app.services.generation import ExtractiveGenerator, GenerationService, LLMGenerator  # noqa: E402
from app.services.validation import extract_citations  # noqa: E402
from evaluation.data import RESULTS_DIR, load_eval, read_jsonl, save_result  # noqa: E402
from evaluation.metrics import mean, percentile  # noqa: E402


def step_match(embedder, generated: list[str], reference: list[str], threshold: float) -> tuple[float, float]:
    if not generated or not reference:
        return 0.0, 0.0
    g = embedder.encode([extract_citations(s)[0] for s in generated])
    r = embedder.encode(reference)
    sims = g @ r.T
    recall = float((sims.max(axis=0) >= threshold).mean())
    precision = float((sims.max(axis=1) >= threshold).mean())
    return recall, precision


async def judge(llm, complaint, steps, reference) -> dict | None:
    try:
        return await llm.generate_json(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": build_judge_user_prompt(complaint, steps, reference)},
            ],
            temperature=0.0,
            max_tokens=1500,  # reasoning judges spend part of the budget before emitting JSON
        )
    except LLMUnavailable:
        return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", choices=["llm", "extractive"], default="extractive")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--match-threshold", type=float, default=0.6)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="judge with a different model than the generator (reduces self-preference bias); same endpoint",
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="judge on a different OpenAI-compatible endpoint (key from env JUDGE_API_KEY), e.g. Groq while generating on Gemini",
    )
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between requests (free-tier rate limits)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start over")
    parser.add_argument(
        "--novel-only", action="store_true", help="only run the novel-class (wave-2) complaints, no generation cases"
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="no new pipeline/LLM calls: summarise the cases already in the checkpoint (e.g. after a quota cap)",
    )
    args = parser.parse_args()
    name = f"generation_{args.generator}{('_' + args.tag) if args.tag else ''}"

    # Checkpointing: each finished case is appended to a JSONL file so a killed run (e.g. by an OS memory
    # guard or a network outage) resumes where it stopped instead of re-spending LLM quota.
    ckpt = RESULTS_DIR / f"{name}.checkpoint.jsonl"
    done: dict[str, dict] = {}
    if ckpt.exists() and not args.fresh:
        # Rows that failed with a generation error (rate limit, outage) are retried, not treated as done.
        done = {r["id"]: r for r in read_jsonl(ckpt) if not r.get("generation_error")}
        print(f"resuming: {len(done)} cases already in {ckpt.name}")
    elif ckpt.exists():
        ckpt.unlink()

    def checkpoint(row: dict) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with ckpt.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        done[row["id"]] = row

    container = pipeline = None
    if args.summarize_only:
        pass  # no models, no DB, no LLM: summarise checkpointed rows only
    elif args.generator == "extractive":
        container = await build_container()
        pipeline = container.pipeline
        pipeline.generation = GenerationService(None, ExtractiveGenerator())
    else:
        container = await build_container()
        pipeline = container.pipeline
        if container.llm is None or not container.llm.available:
            raise SystemExit("LLM not configured: set LLM_API_KEY (and LLM_BASE_URL / LLM_MODEL) in .env")
        pipeline.generation = GenerationService(LLMGenerator(container.llm), ExtractiveGenerator(), fallback=False)

    # Reference-step matching always uses the metric model (MiniLM, cosine >= --match-threshold), independent of the
    # system's embedding model, so the metric stays comparable across embedding switches.
    metric_embedder = None if args.summarize_only else get_embedding_service(settings.METRIC_EMBEDDING_MODEL)

    judge_llm = None
    if args.judge and not args.summarize_only:
        judge_llm = (
            OpenAICompatibleProvider(
                api_key=os.environ.get("JUDGE_API_KEY") or None,
                base_url=args.judge_base_url,
                model=args.judge_model,
                reasoning_effort="low",
            )
            if args.judge_model
            else (container.llm if container is not None else None)
        )
        if judge_llm is None or not judge_llm.available:
            raise SystemExit("--judge needs LLM_API_KEY")

    cases = load_eval("generation_eval")[: args.limit]
    for case in cases:
        if case["id"] in done or args.summarize_only or args.novel_only:
            continue
        assert pipeline is not None and container is not None  # not in --summarize-only mode
        r = await pipeline.process(case["complaint"], record=False)
        recall, precision = step_match(
            metric_embedder, r.generation.resolution_steps, case["reference_steps"], args.match_threshold
        )
        row = {
            "id": case["id"],
            "citation_accuracy": r.validation.citation_accuracy,
            "citation_coverage": r.validation.citation_coverage,
            "groundedness": r.validation.groundedness_score,
            "reference_step_recall": recall,
            "step_precision": precision,
            "n_steps": len(r.generation.resolution_steps),
            "kb_retrieved": float(case["reference_kb_id"] in [d.id for d in r.retrieval.documents]),
            "intent_correct": float(r.metadata.intent == case["category"]),
            "confidence": r.decision.heuristic_confidence,
            "decision": r.decision.decision,
            "latency_ms": r.latency_ms,
            "stage_ms": r.stage_ms,
            "generation_error": r.generation.error,
            "contradictions": sum(c.support_status == "contradicted" for c in r.validation.claims),
            "steps": r.generation.resolution_steps,
            "summary": r.generation.summary,
            "claim_status": [c.support_status for c in r.validation.claims],
        }
        if judge_llm is not None:
            row["judge"] = await judge(
                judge_llm, case["complaint"], r.generation.resolution_steps, case["reference_steps"]
            )
        checkpoint(row)
        if args.delay:
            await asyncio.sleep(args.delay)
    rows = [done[c["id"]] for c in cases if c["id"] in done]
    if not rows and not args.novel_only:
        raise SystemExit("no completed cases to summarise")

    good = [r["reference_step_recall"] >= 0.5 for r in rows]
    conf = [r["confidence"] for r in rows]
    auc = roc_auc_score(good, conf) if 0 < sum(good) < len(good) else None
    by_decision = {}
    for d in ("RESOLVE", "REVIEW", "ESCALATE"):
        sub = [r for r in rows if r["decision"] == d]
        by_decision[d] = {
            "n": len(sub),
            "mean_reference_step_recall": round(mean([r["reference_step_recall"] for r in sub]), 4) if sub else None,
            "mean_groundedness": round(mean([r["groundedness"] for r in sub]), 4) if sub else None,
        }

    novel = load_eval("novel_intent_eval")[: (args.limit or 60)]
    for case in novel:
        if case["id"] in done or args.summarize_only:
            continue
        assert pipeline is not None and container is not None  # not in --summarize-only mode
        r = await pipeline.process(case["complaint"], record=False)
        checkpoint(
            {
                "id": case["id"],
                "decision": r.decision.decision,
                "decision_reason": r.decision.decision_reason,
                "intent": r.metadata.intent,
                "n_steps": len(r.generation.resolution_steps),
                "groundedness": r.validation.groundedness_score,
                "max_relevance": max(r.retrieval.relevance_scores) if r.retrieval.items else 0.0,
                "confidence": r.decision.heuristic_confidence,
                "generation_error": r.generation.error,
            }
        )
        if args.delay:
            await asyncio.sleep(args.delay)
    novel_rows = [done[c["id"]] for c in novel if c["id"] in done]
    novel_decisions = Counter(r["decision"] for r in novel_rows)

    lat = [r["latency_ms"] for r in rows]
    stage_keys = rows[0]["stage_ms"].keys() if rows else []
    summary = {
        "generator": args.generator if args.generator == "extractive" else settings.LLM_MODEL,
        "n": len(rows),
        "n_planned": len(cases),
        "n_novel": len(novel_rows),
        "match_threshold": args.match_threshold,
        "metrics": {
            k: round(mean([r[k] for r in rows]), 4)
            for k in [
                "citation_accuracy",
                "citation_coverage",
                "groundedness",
                "reference_step_recall",
                "step_precision",
                "n_steps",
                "kb_retrieved",
                "intent_correct",
                "confidence",
            ]
        },
        "generation_errors": sum(1 for r in rows if r["generation_error"]),
        "drafts_with_contradiction": sum(1 for r in rows if r["contradictions"]),
        "decisions": dict(Counter(r["decision"] for r in rows)),
        "quality_by_decision": by_decision,
        "confidence_quality_auc": round(auc, 4) if auc is not None else None,
        "latency_ms": {
            "p50": round(percentile(lat, 50)),
            "p95": round(percentile(lat, 95)),
            "max": max(lat) if lat else None,
        },
        "stage_latency_ms_mean": {k: round(mean([r["stage_ms"][k] for r in rows])) for k in stage_keys},
        "novel_intent_decisions": dict(novel_decisions),
        "novel_intent_empty_drafts": sum(1 for r in novel_rows if r["n_steps"] == 0),
        "novel_intent_generation_errors": sum(1 for r in novel_rows if r["generation_error"]),
    }
    judged = [r["judge"] for r in rows if r.get("judge")]
    if judged:
        summary["llm_judge_mean"] = {
            k: round(mean([float(j.get(k, 0)) for j in judged]), 3)
            for k in ("relevance", "completeness", "specificity", "correctness")
        }
        summary["llm_judge_n"] = len(judged)
        summary["llm_judge_model"] = args.judge_model or settings.LLM_MODEL
    save_result(name, {"summary": summary, "rows": rows, "novel_rows": novel_rows})

    print(json.dumps(summary, indent=2))
    if container is not None:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
