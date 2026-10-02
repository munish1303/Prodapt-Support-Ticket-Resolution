# Telecom Support Ticket Resolution Assistant

Prodapt AI Engineering evaluation, **Use Case 2: Intelligent Support Ticket Resolution Assistant**.

An agent pastes a raw customer complaint, for example:

> *"My broadband drops every evening around 8 and I've already restarted the router twice, I work from home and
> this is costing me"*

and gets back:

* **Understanding**: intent `connectivity_issue`, products `[router, broadband]`, severity `high`, sentiment `negative`
* **Sources**: the most similar *resolved* tickets and KB articles (hybrid pgvector + PostgreSQL full-text search)
* **A step-by-step resolution** where every step cites the sources it came from, e.g. `Move the 2.4 GHz radio to the least congested channel [1][4]`
* **Validation** of every step: is the citation valid, and is the step entailed (or contradicted) by the cited source?
* **An evidence-derived confidence score** (not LLM self-assessment, not a probability) and a decision:
  `RESOLVE` / `REVIEW` / `ESCALATE`

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | diagram, components, design decisions + rejected alternatives, deviations from plan, production considerations |
| [EVALUATION.md](EVALUATION.md) | datasets, leakage checks, understanding / retrieval / generation / system metrics, baseline comparison |
| [EXPERIMENTS.md](EXPERIMENTS.md) | E1 retrieval methods, E2 reranking, E3 groundedness methods; hypotheses, procedure, measured results, decisions |
| [docs/api.md](docs/api.md) | endpoint reference with examples (interactive docs at `/docs`) |
| [data/DATASET_STATISTICS.md](data/DATASET_STATISTICS.md) | actual dataset sizes, provenance, splits |

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

> **Windows note:** Docker Desktop stores images on C: by default. The image is ~3 GB (CPU torch + baked models).
> If C: is short on space, set Docker Desktop → Settings → Resources → Advanced → *Disk image location* to another drive.

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
pytest --cov=app                 # unit + API tests run offline (fake embedder/NLI); no DB needed
pytest -m db                     # DB integration tests (needs the database running)
black --check . && flake8
```

## Evaluations and experiments

```bash
python experiments/e3_groundedness.py                       # E3, no DB needed
python evaluation/understanding_eval.py --index pg          # intent / product / severity / sentiment
python experiments/e1_retrieval.py                          # E1: keyword vs lexical vs semantic vs hybrid
python experiments/e2_reranking.py                          # E2: cross-encoder reranking decision
python evaluation/generation_eval.py --generator extractive # end-to-end, no LLM
python evaluation/generation_eval.py --generator llm --delay 2   # end-to-end with the LLM (free-tier pacing)
python evaluation/system_eval.py --url http://localhost:8000     # latency / throughput / error rate
python experiments/evolving_classes.py                      # new ticket class arrives: before/after ingestion via the API
```

Every script writes its raw output to `experiments/results/*.json`. The numbers quoted in the docs come from those files.

## Repository layout

```
app/
  api/v1/          routes, request/response models, service container
  core/            database, LLM provider abstraction, prompts, logging
  models/          internal dataclasses, embedding service
  services/        understanding, retrieval, generation, validation, decision, ingestion, pipeline, monitoring
  utils/           text processing, PII redaction, injection heuristics
data/
  taxonomies/      intents.json, products.json
  processed/       generated corpus (tickets, KB; wave-2 = new ticket class)
  evaluation/      held-out eval sets, TREC qrels, leakage report
scripts/           generate_synthetic, verify_splits, init_db, ingest, report_dataset_stats, datagen/scenarios.py
evaluation/        metrics, understanding/retrieval/generation/system evals, keyword baseline
experiments/       E1-E3 (+ evolving classes), results/*.json
tests/             unit/ (offline), integration/ (API with in-memory container; DB tests marked `db`)
migrations/        SQL schema
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
