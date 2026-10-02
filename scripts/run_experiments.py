"""Run the evaluation suite in a safe order (one heavy job at a time).

Usage:
  python scripts/run_experiments.py                 # everything that needs no LLM quota
  python scripts/run_experiments.py --with-llm      # also the LLM generation eval (needs LLM_API_KEY)
  python scripts/run_experiments.py --only e1 e3    # a subset

Steps (name: command):
  leakage      scripts/verify_splits.py
  e3           experiments/e3_groundedness.py                      (no DB)
  understanding evaluation/understanding_eval.py --index memory   (no DB)
  e1           experiments/e1_retrieval.py                         (DB)
  e2           experiments/e2_reranking.py                         (DB)
  retrieval    evaluation/retrieval_eval.py                        (DB; production retriever config)
  generation   evaluation/generation_eval.py --generator extractive (DB)
  evolving     experiments/evolving_classes.py --extractive        (DB; adds and then removes wave-2 rows)
  llm          evaluation/generation_eval.py --generator llm --judge ... (DB + LLM quota; resumable)
  load         evaluation/system_eval.py (needs the API running on --url)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

STEPS: dict[str, list[str]] = {
    "leakage": ["scripts/verify_splits.py"],
    "e3": ["experiments/e3_groundedness.py"],
    "understanding": ["evaluation/understanding_eval.py", "--index", "memory"],
    "e1": ["experiments/e1_retrieval.py"],
    "e2": ["experiments/e2_reranking.py"],
    "retrieval": ["evaluation/retrieval_eval.py"],
    "generation": ["evaluation/generation_eval.py", "--generator", "extractive"],
    "evolving": ["experiments/evolving_classes.py", "--extractive"],
    "llm": [
        "evaluation/generation_eval.py",
        "--generator",
        "llm",
        "--judge",
        "--judge-model",
        "openai/gpt-oss-120b",
        "--delay",
        "20",
    ],
    "load": ["evaluation/system_eval.py"],
}
DEFAULT = ["leakage", "e3", "understanding", "e1", "e2", "retrieval", "generation", "evolving"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=sorted(STEPS), help="run only these steps")
    parser.add_argument("--with-llm", action="store_true", help="append the LLM generation eval")
    parser.add_argument("--with-load", action="store_true", help="append the load test (API must be running)")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL for the load test")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed step")
    args = parser.parse_args()

    steps = args.only or DEFAULT + (["llm"] if args.with_llm else []) + (["load"] if args.with_load else [])
    failures = []
    for name in steps:
        cmd = [PY, *STEPS[name]] + (["--url", args.url] if name == "load" else [])
        print(f"\n=== {name}: {' '.join(cmd[1:])}", flush=True)
        started = time.perf_counter()
        code = subprocess.call(cmd, cwd=ROOT)
        print(f"=== {name}: exit {code} in {time.perf_counter() - started:.0f}s", flush=True)
        if code != 0:
            failures.append(name)
            if not args.keep_going:
                break
    print("\nresults in experiments/results/; failed steps:", failures or "none")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
