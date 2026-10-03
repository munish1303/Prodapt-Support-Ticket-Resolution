"""Assess the public datasets suggested in the use-case document against our corpus (IMPLEMENTATION_PLAN §12.1).

The use-case document lists two ticket datasets as suggestions ("not restricted to, you can choose your dataset"):

* Hugging Face `Tobi-Bueck/customer-support-tickets` (newest file, aa_dataset-tickets-multi-lang-5-2-50-version.csv)
* GitHub `santhoshmishra/Ticket_data` (data/tickets.csv)

This script downloads them (cached under data/external/, git-ignored because of size and licence), computes the same
text-quality measures on each and on our corpus (data/processed/tickets.jsonl), and writes
experiments/results/public_dataset_assessment.json. The measures are deliberately simple and transparent:

* distinct-3gram: share of distinct word trigrams in a fixed random sample of 2,000 texts (seed 0); higher = more
  varied wording, lower = templated text;
* step language: answer contains a numbered list or one of first/then/next/try/restart/reset/check/update/reinstall/
  clear (a keyword heuristic for "contains an actionable fix");
* asks for info: answer asks the customer to provide/share/send details or "let us know";
* placeholders: answer contains template slots such as <name>, [Name] or {x};
* telecom terms (strict): complaint mentions broadband, fibre line/box, roaming, SIM card, mobile data, landline or
  ONT; (broad): also wifi, router, modem, SIM, 4G/5G/LTE, signal, outage, internet connection, ISP, tariff, DSL.
  Broad terms also occur in office-IT tickets, so the strict list is the better domain signal.

Usage: python scripts/assess_public_datasets.py [--out PATH]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "external"
OURS = ROOT / "data" / "processed" / "tickets.jsonl"

HF_REPO = "Tobi-Bueck/customer-support-tickets"
HF_FILE = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{HF_FILE}"
GH_REPO = "santhoshmishra/Ticket_data"
GH_URL = f"https://raw.githubusercontent.com/{GH_REPO}/main/data/tickets.csv"

PLACEHOLDER = re.compile(r"<[^>]{2,30}>|\[[A-Z][^\]]{1,30}\]|\{[^}]{2,30}\}")
ASKS_INFO = re.compile(
    r"(please (provide|share|send)|could you (please )?(provide|share|send)|can you (provide|share|send)"
    r"|let us know|more (details|information))",
    re.I,
)
STEP_LANGUAGE = re.compile(
    r"(\n\s*\d+[.)]\s|\b(first|then|next|try|restart|reset|check|update|reinstall|clear)\b)", re.I
)
TELECOM = re.compile(r"\b(broadband|fib(re|er) (optic|line|box)|roaming|sim card|mobile data|landline|ont)\b", re.I)
TELECOM_BROAD = re.compile(
    r"\b(broadband|fib(re|er)|router|modem|wi-?fi|sim|roaming|mobile data|4g|5g|lte|signal|outage"
    r"|internet connection|isp|landline|tariff|ont|dsl)\b",
    re.I,
)


def download(url: str, dest: Path, attempts: int = 6) -> Path:
    """Download with retries and resume (the network this was built on drops connections)."""
    if dest.exists() and dest.stat().st_size > 0 and not dest.with_suffix(".part").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                mode = "ab" if have and resp.status == 206 else "wb"
                with open(part, mode) as fh:
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
            part.replace(dest)
            return dest
        except OSError as exc:
            print(f"  download attempt {attempt} failed: {exc}", file=sys.stderr)
            time.sleep(5)
    raise SystemExit(f"could not download {url}")


def fetch_json(url: str) -> dict:
    for _ in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read())
        except (OSError, ValueError):
            time.sleep(3)
    return {}


def read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(10**8)
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def distinct_trigrams(texts: list[str], n: int = 2000, seed: int = 0) -> float:
    sample = random.Random(seed).sample(texts, min(n, len(texts)))
    grams: list[tuple[str, ...]] = []
    for t in sample:
        words = re.findall(r"[a-z']+", t.lower())
        grams.extend(tuple(words[i : i + 3]) for i in range(max(0, len(words) - 2)))
    return round(len(set(grams)) / max(1, len(grams)), 3)


def rate(texts: list[str], pattern: re.Pattern) -> float:
    return round(sum(bool(pattern.search(t)) for t in texts) / max(1, len(texts)), 3)


def top(values: list[str], k: int = 10) -> dict[str, int]:
    return dict(collections.Counter(v.strip() or "(blank)" for v in values).most_common(k))


def measure(complaints: list[str], answers: list[str]) -> dict:
    filled = [a for a in answers if a.strip()]
    return {
        "rows_measured": len(complaints),
        "complaint_median_chars": int(statistics.median(len(c) for c in complaints)),
        "distinct_complaints_share": round(len(set(complaints)) / len(complaints), 3),
        "complaint_distinct_3gram": distinct_trigrams(complaints),
        "complaints_with_telecom_terms_strict": rate(complaints, TELECOM),
        "complaints_with_telecom_terms_broad": rate(complaints, TELECOM_BROAD),
        "answers_filled": len(filled),
        "unique_answers": len(set(filled)),
        "answer_distinct_3gram": distinct_trigrams(filled),
        "answers_with_step_language": rate(filled, STEP_LANGUAGE),
        "answers_asking_for_more_info": rate(filled, ASKS_INFO),
        "answers_with_placeholders": rate(filled, PLACEHOLDER),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "public_dataset_assessment.json"))
    args = ap.parse_args()

    ours = [json.loads(line) for line in OURS.read_text(encoding="utf-8").splitlines() if line.strip()]
    hf_rows = read_csv(download(HF_URL, CACHE / "hf_customer_support_tickets_v5.csv"))
    gh_rows = read_csv(download(GH_URL, CACHE / "github_ticket_data.csv"))
    hf_en = [r for r in hf_rows if (r.get("language") or "").strip() == "en"]

    hf_meta = fetch_json(f"https://huggingface.co/api/datasets/{HF_REPO}")
    gh_meta = fetch_json(f"https://api.github.com/repos/{GH_REPO}")
    hf_license = next((t.split(":", 1)[1] for t in hf_meta.get("tags", []) if t.startswith("license:")), None)

    report: dict[str, Any] = {
        "method": "see module docstring of scripts/assess_public_datasets.py",
        "ours": {
            "what": "synthetic telecom tickets generated from 43 hand-written root-cause scenarios",
            "rows": len(ours),
            "labels": {"category": top([t["category"] for t in ours])},
            "licence": "MIT (this repository)",
            **measure([t["complaint"] for t in ours], [t.get("resolution") or "" for t in ours]),
        },
        "huggingface_tobi_bueck": {
            "what": "synthetic (AI-generated) IT-helpdesk / online-store support emails with agent answers",
            "file": HF_FILE,
            "rows": len(hf_rows),
            "rows_english": len(hf_en),
            "columns": list(hf_rows[0].keys()),
            "labels": {
                "language": top([r["language"] for r in hf_rows]),
                "queue": top([r["queue"] for r in hf_rows]),
                "type": top([r["type"] for r in hf_rows]),
                "priority": top([r["priority"] for r in hf_rows]),
            },
            "licence": hf_license,
            "measured_on": "English rows; complaint = subject + body, answer = agent answer",
            **measure([f"{r['subject'] or ''} {r['body'] or ''}" for r in hf_en], [r["answer"] or "" for r in hf_en]),
        },
        "github_santhoshmishra": {
            "what": "NYC 311-style city service requests (noise, parking, rodents, heat), no free-text complaint",
            "rows": len(gh_rows),
            "columns": list(gh_rows[0].keys()),
            "labels": {
                "agency": top([r["agency"] for r in gh_rows]),
                "complaint_type": top([r["complaint_type"] for r in gh_rows]),
            },
            "licence": (gh_meta.get("license") or {}).get("spdx_id") if gh_meta.get("license") else None,
            "repo_created_at": gh_meta.get("created_at"),
            "measured_on": "complaint = complaint_type + descriptor, answer = resolution_description",
            **measure(
                [f"{r['complaint_type']} {r['descriptor']}" for r in gh_rows],
                [r["resolution_description"] or "" for r in gh_rows],
            ),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    keys = [
        ("rows_measured", "rows measured"),
        ("complaint_median_chars", "complaint median chars"),
        ("distinct_complaints_share", "distinct complaints"),
        ("complaint_distinct_3gram", "complaint distinct-3gram"),
        ("complaints_with_telecom_terms_strict", "complaints with telecom terms (strict)"),
        ("complaints_with_telecom_terms_broad", "complaints with telecom terms (broad)"),
        ("unique_answers", "unique answers"),
        ("answers_with_step_language", "answers with step language"),
        ("answers_asking_for_more_info", "answers asking for more info"),
        ("answers_with_placeholders", "answers with placeholders"),
    ]
    names = ["ours", "huggingface_tobi_bueck", "github_santhoshmishra"]
    print("| measure | ours | HF Tobi-Bueck (EN) | GitHub Ticket_data |")
    print("|---|---:|---:|---:|")
    for key, label in keys:
        print(f"| {label} | " + " | ".join(str(report[n][key]) for n in names) + " |")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
