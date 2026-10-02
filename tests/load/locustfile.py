"""Locust load test for the resolution API.

Interactive:   locust -f tests/load/locustfile.py --host http://localhost:8000
Headless:      locust -f tests/load/locustfile.py --host http://localhost:8000 --headless -u 4 -r 1 -t 60s

Simulated agents mostly resolve complaints (held-out eval phrasings, never seen by the corpus) and occasionally
check health and metrics. evaluation/system_eval.py is the scripted equivalent used for the numbers in EVALUATION.md.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

_EVAL = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "test.jsonl"
COMPLAINTS = [json.loads(line)["complaint"] for line in _EVAL.read_text(encoding="utf-8").splitlines() if line.strip()]


class SupportAgent(HttpUser):
    wait_time = between(1, 3)  # an agent reads the draft before the next ticket

    @task(10)
    def resolve(self) -> None:
        with self.client.post(
            "/api/v1/tickets/resolve",
            json={"complaint": random.choice(COMPLAINTS)},
            name="POST /tickets/resolve",
            timeout=120,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            body = resp.json()
            if body["decision"]["action"] not in {"RESOLVE", "REVIEW", "ESCALATE"}:
                resp.failure("unexpected decision")

    @task(1)
    def health(self) -> None:
        self.client.get("/api/v1/health", name="GET /health")

    @task(1)
    def metrics(self) -> None:
        self.client.get("/api/v1/metrics", name="GET /metrics")
