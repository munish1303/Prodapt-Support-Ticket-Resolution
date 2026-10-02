# Deployment

## Docker Compose (single host)

```bash
cp .env.example .env                         # add LLM settings (optional)
docker compose up --build -d
docker compose logs -f api                   # "startup complete" after about 40 s
curl -s localhost:8000/api/v1/health
```

What happens on start (`scripts/docker_entrypoint.sh`):
1. `init_db.py` applies the schema (idempotent).
2. If `tickets` is empty and `AUTO_INGEST` is not `false`, the corpus is generated (if missing) and ingested,
   and ivfflat indexes are built from the real row count.
3. uvicorn starts one worker on port 8000. The container health check polls `/api/v1/health`.

The image (about 3.7 GB) contains CPU-only PyTorch and all four models; with `HF_HUB_OFFLINE=1` (set in compose) the
container never contacts the Hugging Face Hub. The image runs as a non-root user.

Verified on the dev machine: image build, schema application, existing-corpus detection, model loading (40.7 s),
health check, and an end-to-end `/tickets/resolve` call with the LLM.

## Environment

All configuration comes from environment variables (see docs/setup.md §4). Secrets (`LLM_API_KEY`) belong in `.env`
locally, and in the platform's secret store in production, never in the image or the repository.

## Sizing (measured, see EVALUATION.md §6.1)

* One worker sustains about 1 request/s on a 4-core laptop CPU without the LLM (p50 1.0 s at concurrency 1).
  Throughput does not rise with concurrency inside one worker: inference is CPU-bound and serialised.
* Each worker holds about 1.7 GB of models; PostgreSQL needs well under 1 GB at this corpus size.
* With an LLM, add the provider's latency (measured p50 0.97 s for generation) and stay within its rate limits.

## Scaling path

1. **More workers / replicas** behind a load balancer (the API is stateless; models load per process).
2. **Batched inference service** (GPU) for NLI and embeddings, the dominant CPU cost; also cuts latency.
3. **Database:** keep pgvector up to about 10⁶ vectors (retune `lists`, or switch to HNSW), add read replicas and pgBouncer;
   beyond that, move vectors to a dedicated ANN service with PostgreSQL as system of record.
4. **Reranking** (E2) becomes affordable on GPU: re-run E2 there before enabling `USE_RERANKING`.

## Production checklist (not in this build)

* AuthN/Z at an API gateway (OAuth2/JWT) and per-client rate limits.
* TLS termination, encryption at rest, retention policy for `resolution_requests`.
* Metrics export to Prometheus/Grafana (the `/metrics` JSON already has the numbers) and alerts on the drift report.
* Backups / point-in-time recovery for PostgreSQL.
* A paid LLM tier (free tiers cap at about 100 drafts/day on Groq and 20 requests/day per model on Gemini).
