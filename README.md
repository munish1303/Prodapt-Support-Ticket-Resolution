# Telecom Support Ticket Resolution Assistant

Prodapt AI Engineering evaluation, **Use Case 2: Intelligent Support Ticket Resolution Assistant**.

**Live demo: https://resolve-support-assistant.vercel.app** (Vercel container + Neon Postgres, deployed
automatically from `main`; the first visit after 5 idle minutes waits ~15 s while the models load).

**Promo video**

[![Resolve promo: a "charged twice" complaint gets cited steps, every step is validated, and it resolves at 96 confidence](docs/media/resolve-promo-preview.gif)](docs/media/resolve-promo.mp4?raw=true)

▶ **[Watch the full 56-second promo](docs/media/resolve-promo.mp4?raw=true)** (MP4 download, 19 MB; 1080p, 60 fps, with sound). Made in code with
Remotion and a synthesized soundtrack; the complaint, tickets and scores on screen come from a real response of the
running system.

**Screenshots** (live site; click any image for full size)

<table>
  <tr>
    <td width="50%" valign="top"><a href="docs/screenshots/01-console.png"><img src="docs/screenshots/01-console.png" alt="Agent console"></a><br><sub><b>Agent console</b>: paste a raw customer complaint</sub></td>
    <td width="50%" valign="top"><a href="docs/screenshots/02-result-resolve.png"><img src="docs/screenshots/02-result-resolve.png" alt="RESOLVE result"></a><br><sub><b>RESOLVE at 96</b>: intent, product, severity and sentiment, with a cited, validated draft</sub></td>
  </tr>
  <tr>
    <td valign="top"><a href="docs/screenshots/03-result-citations.png"><img src="docs/screenshots/03-result-citations.png" alt="Cited steps and sources"></a><br><sub><b>Citations</b>: every step cites its sources and is checked against them; the four confidence components</sub></td>
    <td valign="top"><a href="docs/screenshots/04-result-review.png"><img src="docs/screenshots/04-result-review.png" alt="REVIEW result"></a><br><sub><b>REVIEW</b>: critical severity always gets a human check, even at 98 confidence</sub></td>
  </tr>
  <tr>
    <td valign="top"><a href="docs/screenshots/05-result-escalate.png"><img src="docs/screenshots/05-result-escalate.png" alt="ESCALATE result"></a><br><sub><b>ESCALATE</b>: an unseen issue type (roaming); the LLM declines to draft from sources that don't fit</sub></td>
    <td valign="top"><a href="docs/screenshots/06-retrieval-playground.png"><img src="docs/screenshots/06-retrieval-playground.png" alt="Retrieval playground"></a><br><sub><b>Retrieval playground</b>: semantic, lexical and hybrid results side by side (no LLM)</sub></td>
  </tr>
  <tr>
    <td valign="top"><a href="docs/screenshots/07-knowledge-base.png"><img src="docs/screenshots/07-knowledge-base.png" alt="Knowledge base view"></a><br><sub><b>Knowledge base</b>: corpus statistics, retrieval configuration and the live database indexes</sub></td>
    <td valign="top"><a href="docs/screenshots/08-database-record.png"><img src="docs/screenshots/08-database-record.png" alt="Database record"></a><br><sub><b>Database record</b>: any source opens its row, labels, provenance and stored 768-d embedding</sub></td>
  </tr>
  <tr>
    <td valign="top"><a href="docs/screenshots/09-insights.png"><img src="docs/screenshots/09-insights.png" alt="Insights"></a><br><sub><b>Insights</b>: latency percentiles, decision mix, groundedness and the drift monitor</sub></td>
    <td valign="top"><a href="docs/screenshots/10-api-docs.png"><img src="docs/screenshots/10-api-docs.png" alt="API docs"></a><br><sub><b>API docs</b>: interactive OpenAPI documentation at <code>/docs</code></sub></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><a href="docs/screenshots/11-mobile.png"><img src="docs/screenshots/11-mobile.png" alt="Phone layout" width="280"></a><br><sub><b>Phone layout</b></sub></td>
  </tr>
</table>

An agent pastes a raw customer complaint and gets structured understanding, the most similar resolved tickets and
KB articles, a cited step-by-step resolution, validation of every step, and a decision. Real output from the
containerized service (Groq LLM generator):

> *"The fibre box on the wall has a red LOS light and we have had no internet since this morning. I work from home."*

* **Understanding**: intent `connectivity_issue`, products `[fiber, broadband]`, severity `critical`, sentiment `neutral`
* **Resolution** (each step cites the sources it came from; all steps validated as grounded):
  1. Check the thin fibre cable into the ONT for sharp bends, kinks or pinching and straighten it. `[2][3][4][5][6][7]`
  2. Make sure the green fibre connector is fully clicked into the ONT port without touching the fibre end. `[1]…[7]`
  3. Check the outage map for a fibre fault in the area. `[1]…[8]`
  4. If LOS stays red and there is no area-wide outage, raise a fibre fault and dispatch a field engineer. `[4][6][7]`
* **Decision**: `REVIEW` (critical severity always gets a human check), heuristic confidence 0.99, groundedness 1.0, 7.1 s

It does not always get it right: the problem statement's own example complaint gets a fluent, fully cited answer from
the *wrong* scenario. That case is analysed in [EVALUATION.md §6.3](EVALUATION.md).

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | diagram, components, design decisions + rejected alternatives, deviations from plan, production considerations |
| [EVALUATION.md](EVALUATION.md) | datasets, leakage checks, understanding / retrieval / generation / system metrics, baseline comparison |
| [EXPERIMENTS.md](EXPERIMENTS.md) | E1 retrieval methods, E2 reranking, E3 groundedness methods; hypotheses, procedure, measured results, decisions |
| [docs/api.md](docs/api.md) | endpoint reference with examples (interactive docs at `/docs`) |
| [data/DATASET_STATISTICS.md](data/DATASET_STATISTICS.md) | actual dataset sizes, provenance, splits |
| [docs/setup.md](docs/setup.md) · [docs/deployment.md](docs/deployment.md) · [docs/troubleshooting.md](docs/troubleshooting.md) | local setup and config reference · deployment, sizing, scaling · problems actually hit and their fixes |

## Results at a glance

All measured on held-out data (synthetic telecom tickets; eval phrasings never appear in the corpus). Details and
caveats in [EVALUATION.md](EVALUATION.md) and [EXPERIMENTS.md](EXPERIMENTS.md).

| What | Result |
|---|---|
| Finding a highly relevant past ticket in the top 5 | keyword search (status quo) **40.5%** → hybrid retrieval **89.0%** |
| Embedding model (E4, switched) | all-mpnet-base-v2 replaced all-MiniLM-L6-v2: retrieval nDCG@10 0.572 → **0.729**, Recall@10 0.558 → **0.726** (now above the 0.70 target), intent macro-F1 0.723 → **0.889** |
| Hybrid vs semantic-only retrieval | nDCG@10 +0.014 (p = 0.0005), MRR +0.041 (p = 0.003) with weights 0.9/0.1 chosen on dev; reranking now adds only +0.010 P@5 for +2 s, so kept off (E2) |
| Groundedness check, F1 at catching unsupported/contradicted steps | **0.918** multi-method vs 0.694 similarity-only (E3); validation keeps MiniLM: same quality as mpnet (0.927, n.s.) at 1/6 of the cost |
| Understanding (500 complaints) | intent macro-F1 **0.889** · products F1 0.814 · severity acc 0.884 · sentiment acc 0.790 |
| New ticket class, no retraining | 0% → **88%** recognised after ingestion; caught beforehand by the intent-mix drift alert (E4′) |
| LLM drafts (Groq `qwen3.8-27b`, all 100 cases) | reference-step recall 0.765 → **0.865** and step precision 0.626 → **0.736** after the embedding switch (both p ≤ 0.003) · groundedness 0.988 · citation accuracy 0.99 · step precision +0.112 vs extractive (p < 0.001) · confidence AUROC 0.85 |
| Draft quality, 50 drafts rated by an AI rater (Claude, blind to system outputs; **not human**; ratings reviewed and endorsed by the project owner) | **84%** safe to use as-is (72% before the switch); 81% of system RESOLVEs safe (76%); every unsafe draft left is a confusable-scenario draft auto-resolved at confidence 0.83-0.91, which confidence cannot catch (AUROC 0.63, EVALUATION.md §6.4) |
| Unseen issue type (60 complaints) | confident wrong answers with extractive drafts: 52% (48% before the switch: better similarity makes unseen complaints look *more* familiar); with the LLM, which declines when sources don't fit, **28%** |
| Latency / load (1 worker, laptop CPU) | p50 0.9 s / p95 1.6 s without LLM · 1.34 req/s at concurrency 8 (0.84 before the switch, thanks to the validation cache), 0% errors · with the free-tier LLM: p50 8.1 s / p95 **14.7 s** (16.7 s before; target 15 s) |
| Known failure mode | "grounded but wrong": a fluent, well-cited draft from the wrong scenario, documented with root cause (EVALUATION.md §6.3); a runtime detector was built and tested, and not adopted because it cost 16 points of RESOLVE rate (EXPERIMENTS.md E9) |
| Deployment | `docker compose up --build` verified end to end on the dev machine (offline model loading; changing the embedding model re-embeds the corpus automatically on startup) |

## Features

* **Complaint understanding**: intent (k-NN over labelled history, new classes usable as soon as their tickets are
  ingested), products (aliases + implied by similar tickets), severity (similar tickets + urgency/impact cues),
  sentiment (calibrated RoBERTa).
* **Hybrid retrieval**: pgvector cosine + PostgreSQL full-text search fused with weighted RRF; guaranteed KB slots;
  optional metadata filters; optional cross-encoder reranking (off by default, per Experiment 2).
* **Cited RAG generation** through any OpenAI-compatible LLM (Groq, Gemini, OpenAI…), PII redacted before the call,
  prompt-injection hardening, and a deterministic extractive fallback when the LLM is unavailable.
* **Validation**: citation validity plus multi-method groundedness (sentence-level NLI entailment and contradiction,
  semantic similarity, lexical overlap, verbatim-quote rule).
* **Evidence-derived confidence and decisions**: RESOLVE / REVIEW / ESCALATE from retrieval relevance, citation
  quality, groundedness and understanding confidence. Never LLM self-assessment.
* **Evolving data**: incremental, idempotent ingestion; KB versioning; runtime registration of new ticket classes.
* **Web console** at http://localhost:8000: paste a complaint and watch retrieval, drafting and validation happen;
  every citation opens the actual database row. A **Knowledge base** view shows the live PostgreSQL + pgvector corpus
  (counts, indexes, any ticket or KB article with its stored embedding) and a retrieval playground that runs semantic,
  lexical and hybrid search side by side, so you can see that answers come from retrieved data. No build step: plain
  HTML/CSS/JS served by the API.
* **Graceful degradation**: if the database, an understanding/validation model or the LLM fails, the agent still
  gets a safe, flagged answer (ESCALATE or REVIEW) instead of an error; a circuit breaker stops requests piling up on
  a dead database (ARCHITECTURE.md §6.1).
* **Operations**: health, metrics (latency percentiles, decision mix, groundedness, fallback rate, agent feedback),
  drift monitoring (unknown-intent rate, nearest-neighbour similarity, intent-mix shift), structured JSON logs with
  request ids, audit log of every resolution.

## Architecture

```mermaid
flowchart LR
    A[Agent UI] -->|POST /tickets/resolve| E[Embed complaint]
    E --> U[Understanding<br/>intent · products · severity · sentiment]
    E --> R[Hybrid retrieval<br/>pgvector + FTS + RRF]
    U --> G[Generation<br/>LLM w/ citations or extractive]
    R --> G
    G --> V[Validation<br/>citations + groundedness]
    V --> D[Decision<br/>confidence → RESOLVE / REVIEW / ESCALATE]
    U <--> PG[(PostgreSQL + pgvector)]
    R <--> PG
    D --> PG
    G <-->|PII-redacted prompt| LLM[(OpenAI-compatible LLM)]
```

One FastAPI process with six service modules (understanding, retrieval, generation, validation, decision,
ingestion). Details, design decisions and rejected alternatives: [ARCHITECTURE.md](ARCHITECTURE.md).

## Technology stack

| Layer | Choice |
|---|---|
| API | Python 3.11, FastAPI, Pydantic v2, uvicorn |
| Database | PostgreSQL 16 + pgvector (ivfflat), PostgreSQL full-text search (tsvector, `ts_rank`), SQLAlchemy 2 async + asyncpg |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` (768-d) for retrieval and understanding (E4); `all-MiniLM-L6-v2` (384-d) for groundedness premise selection, where it is as accurate and 6x faster |
| NLI (groundedness) | `cross-encoder/nli-deberta-v3-small` |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Reranker (optional) | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | any OpenAI-compatible endpoint; evaluated with Groq `qwen/qwen3.8-27b`, `openai/gpt-oss-20b/120b` and Gemini 3 Flash |
| Packaging | Docker (CPU-only torch, models baked in), Docker Compose |
| Quality | pytest + pytest-asyncio + pytest-cov, black, flake8, mypy |

## Dataset

2,043 historical tickets and 40 KB articles generated from 43 hand-written telecom root-cause scenarios, plus a
held-out "wave 2" ticket class for the evolving-data experiment. Evaluation complaints use phrasings that never
appear in the corpus; leakage checks report zero overlap. Provenance and label source are recorded on every row.
Full statistics: [data/DATASET_STATISTICS.md](data/DATASET_STATISTICS.md). Why synthetic, and what that means for
the numbers: [EVALUATION.md §1](EVALUATION.md), including a measured comparison with the two public datasets the
use-case document suggests (`python scripts/assess_public_datasets.py`).

## Quick start (Docker)

```bash
cp .env.example .env          # optional: add LLM_API_KEY (Groq or Gemini free tier)
docker compose up --build     # Postgres+pgvector, schema, data ingestion, API on :8000
```

The first boot generates the synthetic dataset (if missing), embeds and ingests it, and builds the ANN indexes
(a couple of minutes on CPU). Then:

```bash
curl -s localhost:8000/api/v1/health
curl -s -X POST localhost:8000/api/v1/tickets/resolve -H "Content-Type: application/json" \
  -d '{"complaint": "My broadband drops every evening around 8 and I have already restarted the router twice. I work from home and this is costing me."}'
```

Open http://localhost:8000/docs for the interactive API.

Without an `LLM_API_KEY` the service runs in **extractive mode**: steps are selected from the retrieved sources and
cited, with no LLM involved. The response's `resolution.generator` field says which generator produced the draft.

> **Windows note:** Docker Desktop stores images on C: by default. The API image is 3.7 GB (CPU torch + baked models).
> If C: is short on space, set Docker Desktop → Settings → Resources → Advanced → *Disk image location* to another drive.
> On an 8 GB machine, cap the Docker/WSL VM so Windows keeps headroom: `%USERPROFILE%\.wslconfig` with `[wsl2]` / `memory=3GB` (Postgres + API fit comfortably).

## Web console

Open **http://localhost:8000** once the stack is up.

| | |
|---|---|
| ![Console: complaint input](docs/images/console-idle.png) | ![Console: retrieval in progress](docs/images/console-searching.png) |
| **1. Paste a complaint** (or click an example). | **2. Search:** the assistant reads the complaint and pulls similar past tickets and KB articles; the stage list fills in with the real per-stage timings when the response arrives. |
| ![Console: cited resolution](docs/images/console-result.png) | ![Knowledge base and retrieval playground](docs/images/knowledge-base.png) |
| **3. Result:** the complaint, its understanding, the confidence and decision, steps with clickable citations and a per-step support badge (hover for the similarity/NLI signals), and the scrollable citation list with semantic/lexical ranks. | **Knowledge base:** live counts, category mix, retrieval configuration, the ANN/GIN indexes as defined in Postgres, and the retrieval playground (no LLM). Tabs browse tickets, KB articles and the request log. |

"Open in database" on any citation, playground hit or table row opens the stored record, including its provenance
and the first values of its 768-dimensional embedding:

![Stored ticket row with its embedding](docs/images/database-record.png)

Deep links for demos: `/?q=<complaint>` runs a complaint, `/?view=kb&play=<text>` runs the playground,
`/?record=ticket:TKT-000090` opens a row, `/?insights=1` opens metrics and drift, `&cite=N` highlights citation N on a
result, `/?tri=demo` (or `/?tri=0`-`4`) lights a background triangle as if hovered, with the hero text over it
turned white. Add `&static=1` to skip animations.
The console also respects the OS "reduce motion" setting.

## Local development (without Docker for the API)

```bash
python -m venv .venv && .venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

docker compose up -d db                     # just the database
python scripts/generate_synthetic.py        # deterministic (seed 42)
python scripts/verify_splits.py             # leakage checks: corpus vs eval sets
python scripts/init_db.py                   # schema
python scripts/ingest.py                    # embed + ingest + build ivfflat indexes
python scripts/report_dataset_stats.py      # writes data/DATASET_STATISTICS.md
uvicorn app.main:app --reload
```

## Tests and quality

```bash
pytest --cov=app                 # 107 tests; unit + API run offline (fake embedder/NLI), DB tests skip without a DB
pytest -m db                     # DB integration tests only (needs the database running)
black --check . && flake8 && mypy app scripts evaluation experiments tests
locust -f tests/load/locustfile.py --host http://localhost:8000   # optional interactive load test
```

Current status: 107 passed, 88% line coverage of `app/`, black / flake8 / mypy clean.

## Evaluations and experiments

```bash
python experiments/e3_groundedness.py                       # E3, no DB needed
python evaluation/understanding_eval.py --index pg          # intent / product / severity / sentiment
python experiments/e1_retrieval.py                          # E1: keyword vs lexical vs semantic vs hybrid
python experiments/e2_reranking.py                          # E2: cross-encoder reranking decision
python evaluation/generation_eval.py --generator extractive # end-to-end, no LLM
python evaluation/generation_eval.py --generator llm --delay 2   # end-to-end with the LLM (free-tier pacing)
python evaluation/system_eval.py --url http://localhost:8000     # latency / throughput / error rate
python experiments/evolving_classes.py --extractive        # new ticket class arrives: before/after ingestion
python evaluation/retrieval_eval.py                         # deployed retriever configuration
python scripts/run_experiments.py                           # all of the above in a safe order (see --help)
python evaluation/human_eval_analysis.py                    # after filling data/evaluation/human_eval_sheet.xlsx
python evaluation/human_eval_analysis.py --sheet data/evaluation/ai_eval_sheet.xlsx --out experiments/results/ai_rater_eval.json --rater "AI"
python experiments/e4_embeddings.py                         # E4: embedding models (downloads two extra models)
```

Every script writes its raw output to `experiments/results/*.json`. The numbers quoted in the docs come from those files.

## Repository layout

```
app/
  api/v1/          routes, request/response models, service container, corpus (read-only knowledge-base views)
  static/          web console: index.html, css/app.css, js/app.js (no build step)
  core/            database, LLM provider abstraction, prompts, logging
  models/          internal dataclasses, embedding service
  services/        understanding, retrieval, generation, validation, decision, ingestion, pipeline, monitoring
  utils/           text processing, PII redaction, injection heuristics
data/
  taxonomies/      intents.json, products.json
  processed/       generated corpus (tickets, KB; wave-2 = new ticket class)
  evaluation/      held-out eval sets, TREC qrels, leakage report, human_eval_sheet.xlsx
scripts/           generate_synthetic, verify_splits, init_db, ingest, report_dataset_stats, run_experiments,
                   assess_public_datasets,
                   docker_entrypoint.sh, datagen/scenarios.py
evaluation/        metrics, understanding/retrieval/generation/system evals, keyword baseline,
                   build_human_eval_sheet, human_eval_analysis
experiments/       E1-E3, evolving classes, results/*.json (raw output of every number in the docs)
tests/             unit/ (offline), integration/ (API with in-memory container; DB tests marked `db`), load/locustfile.py
docs/              api.md, setup.md, deployment.md, troubleshooting.md, images/ (console screenshots)
migrations/        SQL schema
Dockerfile, docker-compose.yml, requirements.txt, requirements-dev.txt, pyproject.toml, setup.cfg, LICENSE (MIT)
```

## API summary

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/tickets/resolve` | complaint → understanding, sources, cited resolution, validation, confidence, decision |
| POST | `/api/v1/tickets/{request_id}/feedback` | agent rating of a draft (for future confidence calibration) |
| POST | `/api/v1/ingestion` | add/update resolved tickets and KB articles (incremental, versioned) |
| GET/POST | `/api/v1/taxonomy/intents` | list / register ticket classes at runtime |
| GET | `/api/v1/health` | DB, models, LLM status, corpus size |
| GET | `/api/v1/metrics` | volume, decision mix, latency p50/p95/p99, groundedness, fallback rate, feedback |
| GET | `/api/v1/monitoring/drift` | unknown-intent rate, intent-mix shift, nearest-neighbour similarity, alerts |
| GET | `/api/v1/corpus/stats` · `/corpus/tickets[/{id}]` · `/corpus/kb[/{id}]` | read-only knowledge-base views: counts, indexes, rows with provenance and embedding preview |
| GET | `/api/v1/corpus/search?q=` | raw retrieval without the LLM: semantic vs lexical vs hybrid results side by side |
| GET | `/api/v1/requests` | recent resolutions (PII-redacted) with the sources each one retrieved |
| GET | `/` | web console |

## License

[MIT](LICENSE)
