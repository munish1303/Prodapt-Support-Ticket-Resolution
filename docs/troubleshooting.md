# Troubleshooting

Every entry below was actually hit while building and evaluating this project on a Windows 10 laptop
(8 GB RAM, small C: drive, unreliable network).

## Docker / WSL

**Docker build fails: `apt-get update … Clearsigned file isn't valid, got 'NOSPLIT'`.**
The Debian mirror is fetched over plain HTTP and the network returned a corrupted response. The Dockerfile no longer
has an apt layer (the health check uses Python's stdlib), so this cannot recur.

**Docker build fails in `pip install` with `Name or service not known`.**
A short DNS outage exhausted pip's quick retries. The Dockerfile retries the whole pip step and the model-download
step up to 5 times with 20 s pauses; cached layers (e.g. torch) are reused on rebuild.

**C: drive full / Docker images too large.**
Move Docker's data with Docker Desktop → Settings → Resources → Advanced → *Disk image location*. Do **not** move
`%LOCALAPPDATA%\Docker\wsl` by hand and replace it with a junction or symlink: WSL refuses to attach the virtual disk
(`Wsl/Service/CreateInstance/MountDisk/0x800701c0 … untrusted mount point`). If that happens, delete the junction
(`rmdir`, not recursive delete) and restore the original folder.

**Docker engine returns `500 Internal Server Error`; a build fails with `Read-only file system`.**
The host drive holding Docker's virtual disk is full, so the disk inside the VM went read-only (seen when D: hit
100%). Cause here: the Dockerfile used to `chown -R` the model directory after copying the code, which wrote a full
copy of ~2.5 GB of models into a new layer on every rebuild; it no longer does (the app user only needs to read
`/models`). Recovery: free some space on the host drive, `docker desktop restart` (PostgreSQL runs its normal crash
recovery; check `docker logs support-assistant-db-1` for `database system is ready`), then
`docker image prune -f` and `docker builder prune -f --keep-storage 8GB`, and compact the virtual disk
(docs/deployment.md, "Docker disk usage").

**Machine runs out of memory / background jobs get killed.**
Docker's WSL VM can grow to half of RAM and keep it. Cap it with `%USERPROFILE%\.wslconfig`:
```
[wsl2]
memory=3GB
```
then `wsl --shutdown` and restart Docker Desktop. PostgreSQL plus the API container fit in 3 GB. Run heavy jobs
(evaluations, image builds) one at a time.

**`failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine … The system cannot find the file specified`.**
Docker Desktop is not running (it does not auto-start by default). Start Docker Desktop, wait until it reports the
engine is running, then re-run `docker compose up -d`. To avoid this, enable Settings → General → *Start Docker
Desktop when you sign in*.

**`docker: command not found` in Git Bash after installing Docker Desktop.**
Per-user installs put the CLI in `%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin`; open a new terminal or add it
to `PATH`. If `docker version` shows no server, start Docker Desktop (first start may need the licence prompt).

## LLM providers

**`Connection error` / `WinError 64 The specified network name is no longer available`.**
The network advertises IPv6 but drops it intermittently. Set `LLM_FORCE_IPV4=true`. The client also retries
connection resets at the transport level, and requests with exponential backoff (`LLM_MAX_RETRIES`, 6 is a good value
on flaky networks).

**404 `model … does not exist` / `no longer available to new users`.**
Free-tier model catalogues change. List what your key can use (`GET {LLM_BASE_URL}/models`) and set `LLM_MODEL`.
Evaluated working: Groq `qwen/qwen3.8-27b`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`; Gemini
`gemini-3-flash-preview`.

**429 `rate_limit_exceeded … tokens per day (TPD)` (Groq) or `GenerateRequestsPerDayPerProjectPerModel-FreeTier`
(Gemini).** These are daily caps: about 100 drafts/day on Groq (200k tokens), 20 requests/day per Gemini model.
The service falls back to extractive drafts (flag `llm_unavailable_extractive_fallback`). Evaluations checkpoint
every case; re-run the same command later to resume (`--fresh` starts over).

**Reasoning models return empty or invalid JSON.**
Hidden reasoning tokens consume the output budget. Set `LLM_REASONING_EFFORT=low` (gpt-oss, Gemini thinking models).

**503 `model is currently experiencing high demand` (Gemini).**
Temporary on the provider side; the client retries 5xx. Try another model if it persists.

## Python / pip

**`pip install` says `No matching distribution found` for common packages.**
The index download timed out mid-response. Use `pip install --timeout 120 --retries 5 …`.

**Hugging Face downloads fail with `There is not enough space on the disk`.**
Models cache under `.cache/huggingface/` in the project by default (`app/__init__.py` sets `HF_HOME`); point
`HF_HOME` at a drive with space.

**`Warning: You are sending unauthenticated requests to the HF Hub`.** Harmless. In containers it disappears with
`HF_HUB_OFFLINE=1` (models are baked in).

## Behaviour

**A confident, well-cited answer that is about the wrong problem.**
The known "grounded but wrong" failure mode (EVALUATION.md §4.1, §6.3): validation checks draft ↔ sources, not
sources ↔ complaint. Check the cited sources; watch `/api/v1/monitoring/drift` for intent-mix alerts that indicate a
new issue type.

**Every request is ESCALATEd with the flag `retrieval_unavailable` ("Knowledge base unavailable").**
The database is unreachable or too slow, or the circuit breaker opened after repeated failures. Check
`/api/v1/health` (`database`, `database_circuit`) and `docker compose ps`; once the database is back, the next
request after `DB_CIRCUIT_RECOVERY_S` (30 s) closes the circuit automatically.

**The API takes minutes to become healthy after changing `EMBEDDING_MODEL`.**
Expected: `scripts/reembed.py` re-embeds the corpus before the API starts (about 25 minutes for mpnet on 2 CPU
cores; progress is logged as `tickets re-embedded: N`). It resumes if interrupted.

**Everything is escalated with "contradicted" steps.**
Seen once during development when NLI used multi-sentence premises; fixed (EXPERIMENTS.md, E3 v2). If it reappears
with a different NLI model, re-run `experiments/e3_groundedness.py` and re-tune the `GROUNDED_*` thresholds on its
dev split.
