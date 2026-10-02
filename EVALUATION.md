# Evaluation

All numbers are copied from files in `experiments/results/` (and `data/evaluation/leakage_report.json`)
produced by the scripts named in each section. Where a run is incomplete (the LLM evaluation, capped by the
free-tier daily token quota) the coverage is stated explicitly. Nothing is estimated.

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

### 1.1 Metric definitions

| Metric | Definition |
|---|---|
| **Intent accuracy / macro-F1** | share of complaints whose predicted intent equals the label; macro-F1 = unweighted mean of per-class F1 over the 9 known classes (`unknown_intent` predictions count as errors) |
| **False-unknown rate** | share of in-distribution complaints predicted `unknown_intent` |
| **Product micro P / R / F1** | over (complaint, product) pairs: predicted vs labelled product sets, pooled across complaints; *exact match* = predicted set equals labelled set |
| **Severity accuracy / within-one** | exact 4-level match; within-one = predicted level at most one step from the label |
| **Sentiment accuracy / macro-F1** | 3 classes (negative / neutral / positive) |
| **nDCG@10** | graded relevance (2 = same root-cause scenario, 1 = same intent + overlapping product): DCG = Σ (2^g − 1) / log₂(rank + 1) over the top 10, divided by the ideal DCG |
| **MRR** | mean of 1 / rank of the first grade-2 document in the top 20 (0 if none) |
| **P@k** | grade-2 documents in the top k, divided by k |
| **Hit@k** | 1 if any grade-2 document is in the top k |
| **Recall@k (capped)** | \|grade-2 ∩ top-k\| / min(k, \|grade-2\|); plain recall is uninformative when a query has dozens of equally relevant tickets |
| **KB-Recall@k** | 1 if the scenario's KB article is in the top k |
| **Citation accuracy** | valid citations (source number exists) / all citations in a draft |
| **Citation coverage** | steps with at least one valid citation / all steps |
| **Groundedness** | (supported + 0.5 × weakly supported) / all steps; uncited, unsupported and contradicted steps score 0 |
| **Reference-step recall / step precision** | MiniLM cosine ≥ 0.6 between a generated step and a scenario reference fix step: recall = reference steps matched by some generated step; precision = generated steps matching some reference step |
| **F1 (not-supported)** | E3: detection of unsupported ∪ contradicted claims as the positive class |
| **Contradiction recall** | E3: contradicted claims labelled `contradicted` / all contradicted claims |
| **Confidence AUROC** | area under the ROC curve of the heuristic confidence for predicting a *good* draft (reference-step recall ≥ 0.5); 0.5 = no signal |
| **Intent-mix TVD** | ½ Σ \|p_recent(intent) − p_baseline(intent)\| between two request windows |
| **Latency p50 / p95 / p99** | percentiles of end-to-end server time per request (client time in load tests) |
| **Throughput** | completed requests / wall-clock seconds at a fixed concurrency |

Statistical tests: 95% confidence intervals by percentile bootstrap (1,000 resamples); method comparisons by one-sided
paired bootstrap on per-query or per-case differences (2,000 resamples); p-values are reported, not just "better".

## 2. Understanding (L1)

**Script:** `python evaluation/understanding_eval.py` → `experiments/results/understanding_knn_memory.json`.
Run with an exact in-memory k-NN over the same corpus (no DB). The production path uses the same code with
pgvector ivfflat (`--index pg`, not separately re-run). The pipeline-level intent accuracy measured through
pgvector in §4.1 (0.69 on the 100 generation cases) is in line with the exact-search figure.

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

**Results** (200 held-out queries; full table, CIs and significance tests in EXPERIMENTS.md E1):

| Configuration | nDCG@10 | MRR | P@5 | Hit@5 |
|---|---:|---:|---:|---:|
| keyword baseline (today's practice) | 0.223 | 0.270 | 0.210 | 0.405 |
| lexical, `plainto_tsquery` (AND) | 0.000 | 0.000 | 0.000 | 0.000 |
| lexical, OR-of-terms FTS | 0.274 | 0.452 | 0.270 | 0.565 |
| semantic (pgvector) | 0.557 | 0.600 | 0.545 | 0.715 |
| **hybrid RRF 0.8:0.2 (weight chosen on dev)** | **0.572** | **0.657** | **0.561** | **0.760** |
| hybrid + cross-encoder rerank (E2; disabled: +2.7 s) | 0.625 | 0.723 | 0.613 | 0.800 |

**Deployed configuration** (`python evaluation/retrieval_eval.py` → `experiments/results/retrieval_production.json`):
the generator receives **8 sources** (`RETRIEVAL_TOP_K`) with 2 guaranteed KB slots. On the same 200 queries: MRR 0.657,
P@5 0.561, Hit@5 0.760 (unchanged from E1's hybrid), and **the scenario's KB article is in the generator's context for
77% of complaints, vs 19% without KB slots** (E1 hybrid, top 10). The slots trade two ticket positions for the canonical
procedure. nDCG@10 (0.475) and P@10 are lower only because the list is 8 long with 2 KB positions. Latency p50 151 ms.

**Baseline comparison (L3):** keyword search finds a highly relevant ticket in the top 5 for 40.5% of complaints;
the deployed hybrid retriever does for 76.0% (+0.355, p < 0.001). Hybrid's gain over semantic-only is significant
but modest and concentrated at rank 1 (MRR +0.057, p = 0.0005).

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
address the complaint (measured in §4.2: wrong RESOLVEs fall from 48% to 22%); a complaint↔source relevance verifier (e.g. the cross-encoder from E2
applied to the 3–5 cited sources only) is the natural next safeguard; drift monitoring watches the
nearest-neighbour similarity of incoming traffic.

**First-run note.** The first extractive run (before E3 v2) scored groundedness 0.571 and escalated 88/100 drafts
for "contradictions" on verbatim-quoted steps, which exposed the NLI premise issue fixed in E3 v2 (EXPERIMENTS.md).

### 4.2 LLM generator

`experiments/results/generation_llm.json`. Generator `qwen/qwen3.8-27b`, judge `openai/gpt-oss-120b`
(different model family, to reduce self-preference bias), Groq free tier.

**Coverage: 52 of the 100 planned cases** (GEN-0001 to GEN-0052 of the fixed, pre-shuffled eval order; not a
cherry-picked subset). The run hit Groq's free-tier cap of **200,000 tokens per day per model**: each draft costs
about 1.8–2k tokens, so roughly 100 drafts per day, and the budget was partly spent on earlier aborted runs.
Rate-limited cases were retried later (6 of the 52 ran in the API container), never scored as failures.

| Metric (n = 52) | LLM | Extractive (same 52) | Δ, paired bootstrap |
|---|---:|---:|---|
| Reference-step recall | 0.726 [95% CI 0.607–0.830] | 0.711 | +0.015 (p = 0.37, n.s.) |
| **Step precision** | **0.596** | 0.531 | **+0.065 (p = 0.024)** |
| Groundedness | 0.955 | 1.000 | −0.045 (extractive copies verbatim) |
| Citation accuracy / coverage | 0.962 / 0.962 | 1.000 / 1.000 | |
| Steps per draft | 4.65 | 5.00 | |
| Drafts with a contradicted step | 2 | 0 | |
| Empty drafts (LLM declined: sources don't address the complaint) | 2 | n/a | |
| Decisions RESOLVE / REVIEW / ESCALATE | 39 / 8 / 5 | 42 / 9 / 1 | |
| **Confidence AUROC (predicting a good draft)** | **0.802** | 0.751 (on all 100) | |
| LLM-judge 1–5: relevance / completeness / specificity / correctness | 4.02 / 3.94 / 3.87 / 4.06 | n/a | directional only |
| End-to-end latency p50 / p95 | 5.5 s / 15.4 s | 1.0 s / 1.5 s | |
| Generation stage alone p50 / p95 | 0.97 s / 6.8 s | 0 | |
| Mean stage latency: generate / validate | 6.1 s / 3.5 s | 0 / 0.8 s | |

Quality by decision (LLM): **RESOLVE 0.795** reference-step recall (n = 39), **REVIEW 0.719** (n = 8),
**ESCALATE 0.200** (n = 5). The decision layer separates good drafts from bad ones much more sharply with the LLM
than with extractive drafts, because the LLM declines or writes thin drafts when the evidence is poor, and
validation and confidence pick that up.

**Interpretation.**
* On this dataset the LLM's value is **judgment and focus, not recall**: more precise drafts (fewer irrelevant
  steps), occasional refusals when sources don't fit, and better-separated confidence. Recall is statistically tied
  with the extractive generator. Extractive is a deliberately strong baseline here because historical resolutions are
  templated; on messier real tickets the abstractive advantage should grow, but that is untested.
* Paraphrased LLM steps cost validation time (3.5 s vs 0.8 s mean): quoted steps skip NLI, paraphrases don't.
* **Latency: p95 15.4 s is just over the plan's 15 s MVP target, as measured.** The tail is free-tier rate limiting:
  generation alone has p50 0.97 s, but rate-limit retry waits push the mean to 6.1 s (max 186 s). On a paid tier the
  expected end-to-end p95 is roughly retrieval + generation + validation ≈ 0.3 + 2–7 + 3.5 s. That is an estimate
  from the measured stages, not a measurement.
* LLM-judge scores are directional only (single judge, no human calibration).

**Second LLM family (cross-check): Gemini.** `experiments/results/generation_llm_gemini.json`, generator
`gemini-3-flash-preview` (preview model), same judge, run *inside the API container*. Gemini's free tier allows
only **20 requests/day per model**, so it covers 19 cases. On the 19 complaints all three generators share:

| (n = 19) | Gemini 3 Flash | Qwen 3.8 27B | Extractive |
|---|---:|---:|---:|
| Reference-step recall | 0.781 | 0.763 | 0.657 |
| Step precision | 0.573 | 0.550 | 0.474 |
| Groundedness | 0.944 | 0.947 | 1.000 |
| Judge relevance / completeness / specificity / correctness | 4.05 / 4.00 / 4.21 / 4.00 | 4.00 / 3.89 / 4.00 / 3.95 | n/a |

Gemini on its own 19: decisions 11 / 4 / 4 (RESOLVE / REVIEW / ESCALATE), confidence AUROC 0.792, latency p50 7.3 s.
Two LLMs from different families agree closely, which supports the conclusions above. n = 19 is a consistency check,
not a precise estimate.

**Novel-class complaints with an LLM generator: the "grounded but wrong" test.**
`experiments/results/generation_llm_gptoss20b_novel.json`: the same 60 roaming complaints as §4.1 (a class absent
from the corpus), generator Groq `openai/gpt-oss-20b` (low reasoning effort), run in the API container. A different
model from the Qwen/Gemini runs: their daily quotas were exhausted, and gpt-oss-20b has its own.

| Novel-class complaints (n = 60) | RESOLVE | REVIEW | ESCALATE | Drafts declined (no steps) |
|---|---:|---:|---:|---:|
| Extractive generator (§4.1) | **29 (48%)** | 30 | 1 | n/a |
| LLM generator (gpt-oss-20b) | **13 (22%)** | 17 | 30 | **26 (43%)** |

The prompt rule *"if the sources do not address the complaint, return an empty list"* does real work: the LLM
declined 26 of 60, which the pipeline turns into ESCALATE ("insufficient evidence"). Confident wrong answers on an
unseen issue type fall by more than half (48% → 22%). It does not eliminate them: 13 drafts were still RESOLVEd,
so the traffic-level intent-mix drift alert (E4′) remains necessary as the second line of defence.

**Remaining gap:** Qwen covers 52 of the 100 generation cases (free-tier daily token cap); resumable with
`python evaluation/generation_eval.py --generator llm --judge --judge-model openai/gpt-oss-120b --delay 20`.

## 5. Groundedness validation (L1)

See EXPERIMENTS.md E3. Held-out F1 for detecting not-supported claims (v2, multi-method): **0.918**
(accuracy 0.930) vs 0.897 NLI only, 0.739 lexical only and 0.694 similarity only. False alarms: 0% on verbatim-quoted
steps and 15% on paraphrased steps. The first version (chunk premises, max contradiction) scored 0.736 on the same
set, with 40% false alarms on verbatim steps; the generation eval exposed this and drove the v2 design.

## 6. System health (L4)

### 6.1 Load test

**Script:** `python evaluation/system_eval.py --url http://127.0.0.1:8000 --requests 30 --concurrency 1 4 8`
→ `experiments/results/system_load_extractive.json`. One uvicorn worker on a 4-core laptop CPU (no GPU), API run
with `LLM_PROVIDER=extractive` (the Groq daily quota was exhausted; LLM generation latency is reported separately
in §4.2). 30 distinct held-out complaints per level.

| Concurrency | Throughput | Client latency p50 / p95 / p99 | Server latency p50 / p95 | Error rate |
|---:|---:|---|---|---:|
| 1 | 0.96 req/s | 1,023 / 1,232 / 1,282 ms | 1,004 / 1,212 ms | 0% |
| 4 | 0.83 req/s | 4,614 / 5,677 / 5,752 ms | 4,573 / 5,627 ms | 0% |
| 8 | 0.84 req/s | 9,253 / 10,937 / 11,103 ms | 9,188 / 10,887 ms | 0% |

**Reading.** Zero errors under load, but throughput is flat at about 1 request/s and latency grows linearly with
concurrency. One worker serialises CPU-bound inference (embedding, sentiment, NLI), so extra concurrent requests
queue. Server latency ≈ client latency, so the queue is inside the service, not the network. Scaling path, in
order: more workers or pods behind a load balancer (each holds about 1.7 GB of models), then a batched inference service
(GPU) for NLI and embeddings, which also cuts per-request latency. The plan's MVP target (P95 < 15 s) holds up to
concurrency 8 on this laptop *without* the LLM. With the free-tier LLM (§4.2) end-to-end p95 is 15.4 s at
concurrency 1, just over the target, driven by rate-limit waits.

### 6.2 Monitoring and drift

Runtime health is exposed at `/api/v1/health`, `/api/v1/metrics` (volume, decision mix, latency percentiles,
groundedness, fallback rate, feedback) and `/api/v1/monitoring/drift`. The drift monitor was validated end to end
in E4′ (EXPERIMENTS.md): a new ticket class did **not** trip the unknown-intent or similarity alerts, but did trip
the intent-mix alert added as a result (TVD 0.517 vs a size-aware noise threshold of 0.323 for 60-request windows).

### 6.3 Worked failure case: the problem statement's own example

> *"My broadband drops every evening around 8 and I've already restarted the router twice, I work from home and this
> is costing me"*

Understanding is reasonable (products `broadband, router`, severity `high` from the work-from-home cue, sentiment
negative), but intent is `outage_report` and the extractive draft lists outage/maintenance checks, RESOLVEd at
confidence 0.887. The closest scenario is evening Wi-Fi drops (channel interference) or peak-hour congestion; its
KB article was retrieved, but only as source #7.

Root cause, from per-ranker inspection:
* **Lexical search ranks it well** (5 of its top 6 are evening-Wi-Fi or peak-congestion tickets, matching "every evening around 8").
* **Semantic search ranks area-outage tickets first**, and with semantic weight 0.8 it dominates the fusion. Most of the
  complaint is boilerplate shared across all scenarios ("restarted the router twice", "I work from home and this is
  costing me"). A single embedding of the whole complaint is diluted by it, and "broadband drops" embeds closer to
  "lost internet" than to "wifi keeps dropping".
* The system cannot see its own error: the wrong tickets look relevant (cosine 0.67) and the draft is perfectly grounded
  in them, which is the "grounded but wrong" mode of §4.1.

Not tuned away (fitting the system to its demo example would be overfitting). What the evidence suggests instead:
(1) **ranker disagreement as an uncertainty signal**: here the semantic and lexical top-10s barely overlap, a cheap
runtime clue that should lower confidence; (2) sentence-level / late-interaction retrieval so boilerplate cannot
outvote the symptom sentence; (3) the LLM generator, which sees the evening-Wi-Fi KB article in its context and can
decline or hedge; (4) a complaint↔source relevance verifier (cross-encoder) on the cited sources.

### 6.4 Human evaluation (prepared, awaiting ratings)

The plan's Tier-2 human evaluation is set up but **not yet rated**, so no human results are reported.
`data/evaluation/human_eval_sheet.xlsx` (built by `evaluation/build_human_eval_sheet.py`) holds 50 Qwen drafts
(seed 7, both "declined" drafts included, shuffled). Each row has the complaint, the 8 sources the generator saw
(re-retrieved with the production retriever, verified to reproduce the original run for all 50), the cited draft and
the reference fix. The rater fills relevance / completeness / correctness (1–5), "safe to use as-is" and their own
RESOLVE / REVIEW / ESCALATE. System outputs and LLM-judge scores sit on a hidden sheet to avoid anchoring.

Once rated, `python evaluation/human_eval_analysis.py` reports: decision agreement and Cohen's kappa vs the system,
precision of system RESOLVE, Spearman correlations validating the LLM judge, the heuristic confidence and the automatic
reference-recall metric, confidence AUROC for "safe", and a RESOLVE-threshold sweep, the data Experiment 5
(threshold tuning) needs. The workbook's Summary formulas were verified with the `formulas` engine (no errors when
empty; values match an independent computation on a synthetic fill).

## 7. Engineering checks

| Check | Result |
|---|---|
| Unit + API + DB integration tests (`pytest`) | 69 passed (DB tests run against the live pgvector container; they skip if no DB) |
| Line coverage of `app/` | 83% |
| `black --check`, `flake8`, `mypy` (app, scripts, evaluation, experiments, tests: 66 files) | clean |
| Locust load test (`tests/load/locustfile.py`, 2 users, 40 s, containerized API) | 27 requests, 0 failures; resolve p50 1.3 s, p95 2.7 s |
| `docker compose up --build` (full stack) | verified: API image builds (3.71 GB: CPU torch + 4 baked models), container applies the schema, detects the existing corpus, loads models in 40.7 s, passes its health check, and served a cited LLM draft end to end (7.1 s) |
| Container offline start | `HF_HUB_OFFLINE=1`: zero Hugging Face Hub calls at startup (models baked into the image) |
| Build robustness | no apt layer (stdlib health check); whole-step retries for pip and model downloads. Both were added after the build failed on this network's DNS/HTTP glitches |

## 8. Limitations

* Synthetic data; absolute values are optimistic or pessimistic in unknown ways relative to real tickets (§1).
* Labels by construction: no inter-annotator agreement and no human verification.
* **"Grounded but wrong"** (§4.1, §6.3): validation checks draft ↔ sources, not sources ↔ complaint. When retrieval
  picks a semantically close but wrong scenario, the system can RESOLVE confidently. Mitigated at traffic level by the
  intent-mix drift alert (E4′); per-request mitigations (ranker disagreement, relevance verifier) are next steps.
* Intent macro-F1 (0.723) is below the plan's provisional 0.75 target; the LLM classifier
  (`INTENT_CLASSIFIER=llm`) has not been evaluated.
* LLM generation evaluated on 52 of 100 cases with Qwen, 19 with Gemini, and the 60 novel-class cases with
  gpt-oss-20b: free-tier daily caps (200k tokens/day on Groq, 20 requests/day on Gemini) prevented one model from
  covering everything.
* E3 test half is small (100 items); the TVD drift coefficient was calibrated on this dataset's class mix.
* One worker sustains about 1 request/s on a laptop CPU (§6.1); horizontal scaling or GPU inference is needed for volume.
* Heuristic confidence is uncalibrated (AUROC 0.75 extractive / 0.80 LLM for predicting a good draft); the
  feedback endpoint collects the data needed to calibrate it.
