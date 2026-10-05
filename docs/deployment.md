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

The image (about 4.1 GB of layers) contains CPU-only PyTorch and all five models; with `HF_HUB_OFFLINE=1` (set in compose) the
container never contacts the Hugging Face Hub. The image runs as a non-root user.

Verified on the dev machine: image build, schema application, existing-corpus detection, model loading (40.7 s),
health check, and an end-to-end `/tickets/resolve` call with the LLM.

## Vercel (container image + Neon Postgres)

Vercel runs the same image as a container-image Function (beta, all plans): `Dockerfile.vercel` is a byte-for-byte
copy of `Dockerfile` (a unit test enforces it). The database is Neon Postgres (pgvector) from Vercel's Storage tab.
`vercel.json` deploys pushes to `main` only, since every build downloads ~2 GB of models.

One-time setup:
1. Vercel dashboard → Add New → Project → import the GitHub repository (root directory: repository root).
2. Storage → Create → Neon → connect it to the project. This sets `DATABASE_URL`; the app accepts Neon's
   `postgresql://…?sslmode=require` form and its pooled endpoint (`app/core/database.py`, `normalize_database_url`).
3. Project environment variables: `LLM_API_KEY` (secret), `LLM_BASE_URL`, `LLM_MODEL`, `HF_HUB_OFFLINE=1`,
   `STARTUP_DB_TASKS=false` and `PORT=8000`.
4. Load the database once from the local stack (schema, corpus, embeddings and the embedding-model record):
   `docker compose exec db pg_dump -U support -d support -Fc --no-owner --no-privileges > support.dump`, then
   `pg_restore --no-owner --no-privileges -d "<Neon URL>" support.dump`.
5. Redeploy. After that, every push to `main` builds and deploys automatically.

Why `STARTUP_DB_TASKS=false`: a serverless container starts on every cold start, and the entrypoint's migrations,
re-embedding check and corpus count would add seconds each time. With it set, the entrypoint starts uvicorn
directly; the database is prepared once (step 4). After an embedding-model change, re-run step 4 (or
`scripts/reembed.py` against the Neon URL).

What to expect on the Hobby plan (1 vCPU, 2 GB): the API uses ~0.8-1.2 GB with all models loaded. Instances scale
to zero after 5 minutes without traffic, and the next request waits for model loading (52 s measured locally with
`STARTUP_DB_TASKS=false` on 2 cores, likely longer on 1 vCPU). Open the site once before a demo.

## Environment

All configuration comes from environment variables (see docs/setup.md §4). Secrets (`LLM_API_KEY`) belong in `.env`
locally, and in the platform's secret store in production, never in the image or the repository.

## Sizing (measured, see EVALUATION.md §6.1)

* One worker serves about 1 request/s at concurrency 1 without the LLM (p50 1.0 s) and 1.3 requests/s at
  concurrency 8 (p50 5.2 s), with 0% errors. Inference is CPU-bound; the gain under concurrency comes from the
  validation cache (cited sources repeat), not from parallel inference.
* A process with all models loaded (mpnet and MiniLM embeddings, NLI, sentiment) measured about 1.2 GB resident
  (`docker stats`); allow ~2 GB per worker for request-time growth. PostgreSQL needs well under 1 GB here.
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

## Changing the embedding model

Set `EMBEDDING_MODEL` and `EMBEDDING_DIM`, rebuild the image so the model is baked in (add it to the Dockerfile's
model layer), and restart. On startup `scripts/reembed.py` compares the model recorded in `system_meta` with the
configured one; if they differ it resizes the vector columns, re-embeds all tickets and KB articles in committed
batches and rebuilds the ivfflat indexes before the API starts. It can also be run by hand, in time slots:

```bash
docker compose run --rm --no-deps --entrypoint python api scripts/reembed.py --dry-run          # show the plan
docker compose run --rm --no-deps --entrypoint python api scripts/reembed.py --max-minutes 8    # resumable chunk
```

After a model change, re-tune the scale-dependent thresholds on the dev split (EXPERIMENTS.md, E4 re-baseline) and
re-run the evaluations; `experiments/recalibrate_thresholds.py` translates the thresholds that have no tuning
procedure of their own. The drift baseline restarts automatically, because similarities from different models are not
comparable.

## Docker disk usage on a laptop

Docker Desktop keeps images in one virtual disk (`docker_data.vhdx`) that grows but never shrinks on its own. Every
image rebuild that changes a large layer adds a copy of it; old builds stay until pruned. Keep an eye on it with
`docker system df`, prune unused images and build cache with `docker image prune -f` and
`docker builder prune -f --keep-storage 8GB` (keeps the recent cache so model layers are not downloaded again), and
compact the virtual disk to return the freed space to the host drive (Docker Desktop stopped, `wsl --shutdown`, then
`diskpart`: `select vdisk file="...\docker_data.vhdx"`, `attach vdisk readonly`, `compact vdisk`, `detach vdisk`;
needs an administrator prompt).

