# Local development setup

## Prerequisites

* Python 3.10+ (developed on 3.14, container uses 3.11)
* Docker Desktop (for PostgreSQL + pgvector; optional for the API itself)
* About 2 GB RAM for the models (MiniLM, DeBERTa NLI, RoBERTa sentiment) plus about 1 GB for PostgreSQL
* Optional: an API key for any OpenAI-compatible LLM (Groq or Gemini free tiers work)

## 1. Python environment

```bash
python -m venv .venv
.venv/Scripts/activate                       # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt          # runtime + test + lint + load-test tools
```

Models download on first use into `.cache/huggingface/` inside the project (set `HF_HOME` to change it).

## 2. Database

```bash
docker compose up -d db                      # PostgreSQL 16 + pgvector on localhost:5432
python scripts/init_db.py                    # schema (idempotent)
```

## 3. Data

```bash
python scripts/generate_synthetic.py         # deterministic, seed 42
python scripts/verify_splits.py              # leakage checks; exits non-zero on failure
python scripts/ingest.py                     # embeds and ingests, then builds ivfflat indexes sized from row counts
python scripts/report_dataset_stats.py       # writes data/DATASET_STATISTICS.md
```

## 4. Configuration

Copy `.env.example` to `.env`. Every setting in `app/config.py` can be overridden by an environment variable of
the same name. The most useful ones:

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://support:support@localhost:5432/support` | async SQLAlchemy URL |
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` | Groq, `qwen/qwen3.8-27b`, empty | any OpenAI-compatible endpoint; empty key = extractive mode |
| `LLM_PROVIDER` | `openai_compatible` | `extractive` forces the no-LLM generator |
| `LLM_REASONING_EFFORT` | unset | `low` for reasoning models (gpt-oss, Gemini thinking models) |
| `LLM_FORCE_IPV4` / `LLM_MAX_RETRIES` | `false` / `3` | network resilience (see troubleshooting.md) |
| `INTENT_CLASSIFIER` | `knn` | `llm` to classify intent with the LLM |
| `RRF_SEMANTIC_WEIGHT` / `RRF_LEXICAL_WEIGHT` | `0.8` / `0.2` | hybrid fusion weights (chosen in E1) |
| `USE_RERANKING` | `false` | cross-encoder reranking (E2: better quality, +2.7 s on CPU) |
| `DECISION_RESOLVE_THRESHOLD` / `DECISION_REVIEW_THRESHOLD` | `0.75` / `0.55` | provisional decision thresholds |
| `GROUNDED_*`, `EVIDENCE_*`, `CONF_W_*` | see config | validation, sufficiency and confidence weights (all provisional) |
| `DRIFT_*` | see config | drift alert thresholds |

## 5. Run

```bash
uvicorn app.main:app --reload                # http://localhost:8000/docs
```

## 6. Tests and quality

```bash
pytest --cov=app                             # DB tests run when the database is reachable, otherwise skip
black --check . && flake8 && mypy app scripts evaluation experiments tests
```

## 7. Evaluations and experiments

`python scripts/run_experiments.py --help` runs the whole evaluation suite in a safe order, or see the individual
commands in the README. Every script writes JSON to `experiments/results/`.
