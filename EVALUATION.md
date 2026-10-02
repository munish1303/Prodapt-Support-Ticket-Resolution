# Evaluation

All numbers are copied from files in `experiments/results/` (and `data/evaluation/leakage_report.json`)
produced by the scripts named in each section. Sections marked **PENDING** have runnable code but have not been
executed yet (they need the PostgreSQL database and, for LLM rows, an API key). Nothing in them is estimated.

## 1. Data and methodology

**Dataset.** Synthetic, generated from 43 hand-authored telecom root-cause scenarios (40 at launch + 3 for a
later "roaming" class). Full statistics: [data/DATASET_STATISTICS.md](data/DATASET_STATISTICS.md). Corpus:
2,043 tickets + 40 KB articles. Labels are correct *by construction* (`label_source = generator_by_construction`).

**Why synthetic, and what that means.** Public support-ticket datasets are not telecom-specific and lack
root-cause grouping, so they can't provide graded relevance judgments or reference fixes. Generating from
scenarios gives exact relevance labels and reference steps, which makes retrieval and generation measurable.
The cost is that the language is less varied than real tickets, so **absolute values will not transfer to
production; comparisons between methods are the meaningful output.**

**Making the test hard on purpose.** Each scenario has 6 symptom paraphrases. Corpus tickets use paraphrases
1–4 only; every evaluation complaint uses paraphrases 5–6 and a held-out detail sentence. Eval queries therefore
describe the same problem *in words the corpus never uses*, which is precisely the failure mode of keyword search
in the problem statement.

**Leakage checks** (`scripts/verify_splits.py`, run on all eval sets vs the full corpus incl. wave 2):

| set | n | id overlap | exact-text overlap | template-family overlap | near-dups (cos ≥ 0.95) | mean max-cos to corpus |
|---|---:|---:|---:|---:|---:|---:|
| understanding_eval | 500 | 0 | 0 | 0 | 0 | 0.658 |
| retrieval_eval | 200 | 0 | 0 | 0 | 0 | 0.663 |
| generation_eval | 100 | 0 | 0 | 0 | 0 | 0.665 |
| test (dev split) | 199 | 0 | 0 | 0 | 0 | 0.651 |
| novel_intent_eval | 60 | 0 | 0 | 0 | 0 | 0.710 |

**Dev vs reported sets.** Every threshold chosen from data (unknown-intent operating point, product-share
threshold, sentiment polarity thresholds, severity-bump ablation, groundedness thresholds) was selected on the
**dev split** (`test.jsonl`, or the dev half of the E3 set), never on the set the metric is reported on.

## 2. Understanding (L1)

**Script:** `python evaluation/understanding_eval.py` → `experiments/results/understanding_knn_memory.json`.
Run with an exact in-memory k-NN over the same corpus (no DB). The production path uses the same code with
pgvector ivfflat (`--index pg`, PENDING); ANN search may differ slightly from exact search.

Configuration: k-NN intent (k = 15, weight = sim³), min vote share 0.35, min nearest similarity 0.40;
product share threshold 0.6; sentiment polarity thresholds −0.85 / 0.0; severity sentiment bump disabled.

### 2.1 Results on 500 held-out complaints

| Field | Metric | Value | Plan target (provisional) |
|---|---|---:|---:|
| Intent | accuracy | **0.724** | |
| | macro-F1 | **0.723** | > 0.75 (not met) |
| | flagged unknown (false-unknown rate) | 0.026 | |
| | accuracy when not flagged unknown | 0.743 | |
| Products | micro precision / recall / F1 | 0.689 / 0.745 / **0.716** | |
| | exact set match | 0.558 | |
| | primary product recalled | 0.786 | |
| Severity (4 levels) | accuracy | **0.784** | |
| | macro-F1 | 0.768 | |
| | within one level | 0.930 | |
| Sentiment (3 classes) | accuracy | **0.790** | |
| | macro-F1 | 0.742 | |
| Latency | understanding per complaint (CPU, in-memory index) | mean 266 ms, p95 319 ms | |

Per-class intent F1: technical_support 0.855, outage_report 0.830, billing_dispute 0.809, service_activation
0.781, speed_issue 0.746, account_management 0.667, connectivity_issue 0.661, hardware_problem 0.636,
**plan_change 0.528**. The weak classes are the semantically overlapping ones: plan changes vs billing (both
about price), hardware vs connectivity (a rebooting router *looks like* a dropping connection).

Severity confusion (rows = true, cols = predicted; low/medium/high/critical):
`[[105, 8, 4, 2], [6, 150, 13, 15], [7, 10, 59, 31], [0, 7, 5, 78]]`. The main error is high → critical (31),
where urgency cues ("whole street", "no internet at all") override the neighbour-based base level.

### 2.2 What changed during development, and why (all decided on the dev split)

| Change | Dev-split effect | Eval-set effect (before → after) |
|---|---|---|
| Unknown-intent operating point 0.45/0.40 → 0.35/0.40 | false-unknown 9.0% → 2.5%, accuracy 0.688 → 0.719 | intent acc 0.684 → 0.724; false-unknown 0.102 → 0.026 |
| Products implied by same-class neighbours + drop ambiguous aliases (`calls`, `texts`, `account`, `app`) | product F1 0.385 → 0.767 | 0.401 → 0.716 |
| Sentiment: calibrated polarity thresholds instead of argmax | accuracy ≈0.63 → 0.844 | 0.628 → 0.790 |
| Severity: disable plan's "very negative ⇒ bump medium to high" | accuracy 0.558 → 0.749 | 0.590 → 0.784 |

The sentiment change deserves a note. The off-the-shelf model labels almost every complaint negative because
every complaint describes a problem; the label definition here is *expressed frustration*. That is a
label-definition mismatch, fixed by calibration rather than a different model.

### 2.3 Unknown intents and evolving classes

Dev-split sweep of the unknown-intent thresholds (selected rows; `unknown_threshold_sweep_on_dev` in the results file):

| min vote share | min similarity | dev false-unknown | dev accuracy | novel complaints flagged |
|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 0.000 | 0.729 | 0.000 |
| **0.35** | **0.40** | **0.025** | **0.719** | **0.050** |
| 0.45 | 0.40 | 0.091 | 0.688 | 0.133 |
| 0.55 | 0.40 | 0.206 | 0.628 | 0.483 |
| 0.55 | 0.55 | 0.271 | 0.578 | 0.600 |

**Finding: k-NN vote share is a weak novelty detector.** Roaming complaints sit close to real connectivity
and billing tickets, so the vote is confident but wrong. Catching about half of them would wrongly flag about 21%
of normal traffic. The chosen operating point favours accuracy on known classes; novel classes are meant to be
caught downstream (low relevance/groundedness ⇒ ESCALATE; measured in §4 when run) and by the drift monitor.

**Evolving classes, offline** (60 held-out roaming complaints):

| | before the class exists | after registering `roaming_issue` + ingesting 221 wave-2 tickets |
|---|---|---|
| predicted `roaming_issue` | n/a | **85.0%** (51/60) |
| flagged unknown | 5.0% (3/60) | 0% |
| other predictions | connectivity 25, billing 16, outage 15, speed 1 | outage 5, connectivity 4 |
| in-distribution intent accuracy (500 set) | 0.724 | 0.682 |

No retraining or redeploy: the class becomes predictable from ingestion alone. **Cost:** in-distribution accuracy
drops by 0.042 because roaming tickets now attract some mobile complaints of other classes. In production the new
class's tickets should be reviewed before ingestion, and the per-class F1 rechecked after.

## 3. Retrieval (L1) and baseline comparison (L3)

**Script:** `python experiments/e1_retrieval.py` (see EXPERIMENTS.md E1).
Metrics: nDCG@10 (graded), MRR@20, P@5, Hit@5, capped Recall@10 (`|rel ∩ top10| / min(10, |rel|)`, because many
queries have dozens of equally relevant tickets, which makes plain Recall@10 uninformative), KB-Recall@10.

**Results:** PENDING: [TO BE MEASURED]. Keyword baseline, lexical (AND/OR), semantic and hybrid are compared on
the same 200 queries.

## 4. Generation and pipeline (L2)

**Script:** `python evaluation/generation_eval.py --generator extractive|llm` on 100 held-out complaints.
Metrics: citation accuracy, citation coverage, groundedness (validation service), reference-step recall and
step precision (generated vs scenario fix steps, MiniLM cosine ≥ 0.6), scenario KB article retrieved, decision
mix, latency per stage, **AUROC of heuristic confidence for predicting a good draft** (reference-step recall ≥ 0.5),
and the decision mix on 60 novel-class complaints. Optional LLM-as-judge (directional only: LLM judges favour
LLM text and are not a substitute for human review).

### 4.1 Extractive generator (no LLM) ✅

`experiments/results/generation_extractive.json` (100 held-out complaints + 60 novel-class complaints).

| Metric | Value |
|---|---:|
| Citation accuracy / coverage | 1.000 / 1.000 |
| Groundedness | 1.000 (steps are quoted verbatim from sources) |
| Reference-step recall / step precision | 0.743 / 0.560 |
| Steps per draft | 5.0 |
| Scenario KB article among sources | 0.78 |
| Intent correct (pipeline, pgvector ANN) | 0.69 |
| Mean heuristic confidence | 0.871 |
| Decisions (RESOLVE / REVIEW / ESCALATE) | 81 / 18 / 1 |
| Confidence AUROC for predicting a good draft (step recall ≥ 0.5) | **0.751** |
| End-to-end latency p50 / p95 / max | 1,014 / 1,464 / 4,109 ms |
| Mean stage latency: embed / understand+retrieve / validate | 47 / 272 / 769 ms |
| **Novel-class complaints (60): RESOLVE / REVIEW / ESCALATE** | **29 / 30 / 1** |

Quality by decision: RESOLVE drafts recover 0.748 of reference steps, REVIEW 0.708. The confidence score ranks
drafts better than chance (AUROC 0.75) but the RESOLVE/REVIEW gap is small: it is a useful but weak signal,
as expected for an uncalibrated heuristic.

**Finding: "grounded but wrong".** Half of the complaints about a ticket class the system has never seen
(roaming) are RESOLVEd. The drafts are faithfully grounded, just in the wrong tickets: roaming complaints are
similar enough to mobile-connectivity tickets (cosine ≈ 0.7) to pass the relevance gate, and the extractive
generator has no notion of "these sources don't answer this question". Groundedness verifies *draft ↔ sources*,
not *sources ↔ complaint*. Mitigations: the LLM generator is instructed to return no steps when sources don't
address the complaint (measured in §4.2); a complaint↔source relevance verifier (e.g. the cross-encoder from E2
applied to the 3–5 cited sources only) is the natural next safeguard; drift monitoring watches the
nearest-neighbour similarity of incoming traffic.

**First-run note.** The first extractive run (before E3 v2) scored groundedness 0.571 and escalated 88/100 drafts
for "contradictions" on verbatim-quoted steps, which exposed the NLI premise issue fixed in E3 v2 (EXPERIMENTS.md).

### 4.2 LLM generator

PENDING: [TO BE MEASURED], run in progress (generator `qwen/qwen3.8-27b`, judge `openai/gpt-oss-120b`, Groq free tier).

## 5. Groundedness validation (L1)

See EXPERIMENTS.md E3. Held-out F1 for detecting not-supported claims (v2, multi-method): **0.918**
(accuracy 0.930) vs 0.897 NLI only, 0.739 lexical only and 0.694 similarity only. False alarms: 0% on verbatim-quoted
steps and 15% on paraphrased steps. The first version (chunk premises, max contradiction) scored 0.736 on the same
set, with 40% false alarms on verbatim steps; the generation eval exposed this and drove the v2 design.

## 6. System health (L4)

**Script:** `python evaluation/system_eval.py --url http://localhost:8000` (concurrency 1/4/8): client/server
latency p50/p95/p99, throughput, error rate, and the generator actually used (LLM vs fallback).
Runtime health is exposed continuously at `/api/v1/health`, `/api/v1/metrics` and `/api/v1/monitoring/drift`.

**Results:** PENDING: [TO BE MEASURED].

## 7. Engineering checks

| Check | Result |
|---|---|
| Unit + API tests (`pytest`) | 57 passed, 1 skipped (DB tests skip without a database) |
| Line coverage of `app/` | 72% (DB-only modules `retrieval`, `ingestion`, `monitoring` are covered by `tests/integration/test_db.py` once the DB is up) |
| `black --check`, `flake8` | clean |

## 8. Limitations

* Synthetic data; absolute values are optimistic or pessimistic in unknown ways relative to real tickets (§1).
* Labels by construction: no inter-annotator agreement and no human verification.
* Intent macro-F1 (0.723) is below the plan's provisional 0.75 target; the LLM classifier
  (`INTENT_CLASSIFIER=llm`) has not been evaluated yet.
* E3 test half is small (80 triples).
* Heuristic confidence is uncalibrated; the feedback endpoint collects data for that.
