"""System-level load test against a running API: latency percentiles, throughput, error rate.

Usage (API must be running, e.g. `docker compose up`):
  python evaluation/system_eval.py --url http://localhost:8000 --requests 60 --concurrency 1 4 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.data import load_eval, save_result  # noqa: E402
from evaluation.metrics import percentile  # noqa: E402


async def run_level(client: httpx.AsyncClient, url: str, complaints: list[str], concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    latencies, statuses, generators, server_ms = [], Counter(), Counter(), []

    async def one(text: str):
        async with sem:
            t0 = time.perf_counter()
            try:
                resp = await client.post(f"{url}/api/v1/tickets/resolve", json={"complaint": text}, timeout=120)
                statuses[resp.status_code] += 1
                if resp.status_code == 200:
                    body = resp.json()
                    generators[body["resolution"]["generator"]] += 1
                    server_ms.append(body["latency_ms"])
            except httpx.HTTPError as exc:
                statuses[type(exc).__name__] += 1
            latencies.append((time.perf_counter() - t0) * 1000)

    started = time.perf_counter()
    await asyncio.gather(*(one(c) for c in complaints))
    wall = time.perf_counter() - started
    n = len(complaints)
    ok = statuses.get(200, 0)
    return {
        "concurrency": concurrency,
        "requests": n,
        "wall_s": round(wall, 2),
        "throughput_rps": round(n / wall, 3),
        "error_rate": round(1 - ok / n, 4),
        "status_counts": {str(k): v for k, v in statuses.items()},
        "client_latency_ms": {p: round(percentile(latencies, q)) for p, q in (("p50", 50), ("p95", 95), ("p99", 99))},
        "server_latency_ms": (
            {p: round(percentile(server_ms, q)) for p, q in (("p50", 50), ("p95", 95))} if server_ms else None
        ),
        "generators": dict(generators),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    complaints = [c["complaint"] for c in load_eval("test")]
    async with httpx.AsyncClient() as client:
        health = (await client.get(f"{args.url}/api/v1/health", timeout=30)).json()
        await client.post(
            f"{args.url}/api/v1/tickets/resolve", json={"complaint": complaints[0]}, timeout=120
        )  # warm-up
        levels = []
        for i, c in enumerate(args.concurrency):
            batch = complaints[i * args.requests : (i + 1) * args.requests] or complaints[: args.requests]
            levels.append(await run_level(client, args.url, batch, c))
            print(json.dumps(levels[-1]))
    name = "system_load" + (f"_{args.tag}" if args.tag else "")
    print(f"saved {save_result(name, {'health': health, 'levels': levels})}")


if __name__ == "__main__":
    asyncio.run(main())
