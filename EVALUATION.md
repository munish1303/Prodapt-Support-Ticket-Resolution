# Evaluation

All numbers are copied from files in `experiments/results/` (and `data/evaluation/leakage_report.json`)
produced by the scripts named in each section. Where a run is incomplete (the LLM evaluation, capped by the
free-tier daily token quota) the coverage is stated explicitly. Nothing is estimated.

## Plan targets vs measured

The plan's provisional targets (IMPLEMENTATION_PLAN.md §4.2 / §40.2) against what was measured. Missed targets are
reported as missed; the sections below explain each number.

| Target | Measured | Met | Where |
|---|---|:---:|---|
| Intent classification macro-F1 > 0.75 | 0.723 (500 held-out complaints) | ✗ | §2.1 |
| Product extraction F1 > 0.70 | 0.716 | ✓ | §2.1 |
| Hybrid retrieval Recall@10 > 0.70 | 0.558 deployed (capped Recall@10, E1 test split; plain Recall@10 is uninformative when a query has dozens of equally relevant tickets, §1.1). E4: all-mpnet-base-v2 reaches 0.721 semantic-only; switch recommended, not yet deployed | ✗ (deployed) | EXPERIMENTS.md E1, E4 |
| Citation accuracy > 0.90 | 0.950 (LLM, 100 cases; 1.000 on the 95 non-empty drafts, the 5 declines score 0); 1.000 (extractive) | ✓ | §4 |
| Groundedness > 0.80 | 0.942 (LLM, 100 cases); E3 multi-method checker F1 0.918 | ✓ | §4, §5 |
| P95 latency < 15 s (MVP) | 1.2 s without LLM (concurrency 1); 16.7 s with the free-tier LLM, driven by rate-limit waits | ✗ (with LLM) | §4.2, §6.1 |
| Throughput > 5 req/s | ~1 req/s per worker on the laptop CPU, 0% errors at concurrency 8 | ✗ | §6.1 |
| Test coverage > 70% | 88% of `app/` | ✓ | §7 |
| Experiments 1-3 run, with decisions | E1, E2, E3 done; optional E4 (embeddings) and E5 (thresholds, on AI ratings) done; E4′ evolving classes | ✓ | EXPERIMENTS.md |

The misses have known causes and next steps (§9): recall is limited by the embedding model (E4: all-mpnet-base-v2
clears the target; switching needs a re-baseline) and intent by k-NN over the same embeddings (the LLM classifier
is the other lever); throughput and latency by CPU inference in one worker
and free-tier LLM limits (more workers, batched GPU inference, a paid LLM tier).

## 1. Data and methodology

**Dataset.** Synthetic, generated from 43 hand-authored telecom root-cause scenarios (40 at launch + 3 for a
later "roaming" class). Full statistics: [data/DATASET_STATISTICS.md](data/DATASET_STATISTICS.md). Corpus:
2,043 tickets + 40 KB articles. Labels are correct *by construction* (`label_source = generator_by_construction`).

**Why synthetic, and what that means.** Public support-ticket datasets are not telecom-specific and lack
root-cause grouping, so they can't provide graded relevance judgments or reference fixes. Generating from
scenarios gives exact relevance labels and reference steps, which makes retrieval and generation measurable.
The cost is that the language is less varied than real tickets, so **absolute values will not transfer to
production; comparisons between methods are the meaningful output.**

**Public datasets considered (measured).** The use-case document suggests two ticket datasets ("not restricted to,
you can choose your dataset"). `scripts/assess_public_datasets.py` downloads both and measures them the same way as
our corpus (`experiments/results/public_dataset_assessment.json`; measures defined in the script's docstring):

| | Our corpus | HF `Tobi-Bueck/customer-support-tickets` (English rows) | GitHub `santhoshmishra/Ticket_data` |
|---|---|---|---|
| What it is | synthetic telecom tickets from 43 scenarios | synthetic (AI-generated) IT-helpdesk and online-store emails | NYC 311-style city service requests (noise, parking, rodents, heat) |
| Rows | 2,043 | 28,587 in the newest file (16,338 English) | 25,921 |
| Free-text complaint | yes, median 179 chars | yes, median 416 chars | no: category + descriptor; 0.1% distinct |
| Wording variety (distinct word trigrams) | 0.094 | **0.501** | 0.009 |
| Telecom terms, strict / broad list | 10% / 51% (rest are billing and account tickets; telecom by construction) | 0% / 4% | 0% / 0% |
| Answers containing actionable steps | **85%** | 16% | 0% (16 canned outcomes) |
| Answers asking the customer for more information | 0% | 60% | 0% |
| Answers with template placeholders (`<name>`) | 0% | 56% | 0% |
| Licence | MIT | CC-BY-NC-4.0 (non-commercial) | none stated |

*Decision.* The GitHub dataset is not support tickets and has nothing to retrieve or cite. The Hugging Face dataset
has far more varied wording (the weakness of our templated corpus) but is itself synthetic, IT-helpdesk rather than
telecom, and most of its answers acknowledge the problem or ask for details instead of resolving it, so it would give
the generator little to cite. Neither has root-cause groups for graded relevance. We therefore use the scenario-based
corpus for retrieval and evaluation; the Hugging Face data is the natural out-of-domain stress set (Future work, §9).

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

**Coverage: all 100 planned cases** (GEN-0001 to GEN-0100, the fixed pre-shuffled eval order). The run spanned
two days because Groq's free tier caps each model at **200,000 tokens per day** (about 100 drafts). Every
rate-limited case was retried later rather than scored as a failure, and one case whose understanding stage hit an
early version of the database timeout (ARCHITECTURE.md §6.1) was discarded and re-run, so no case was scored in a
degraded state. The extractive generator ran on the same 100 complaints, so the comparison below is paired.

| Metric (n = 100, paired) | LLM | Extractive | Δ, paired bootstrap |
|---|---:|---:|---|
| Reference-step recall | 0.765 [95% CI 0.695-0.831] | 0.743 | +0.022 (p = 0.20, n.s.) |
| **Step precision** | **0.626** | 0.560 | **+0.066 (p = 0.002)** |
| Groundedness | 0.942 (0.991 over the 95 non-empty drafts) | 1.000 | -0.058 (extractive copies verbatim) |
| Citation accuracy / coverage | 0.950 / 0.950 (every one of the 95 non-empty drafts cites validly; the 5 declined drafts score 0) | 1.000 / 1.000 | |
| Steps per draft | 4.54 | 5.00 | |
| Drafts with a contradicted step | 3 | 0 | |
| Empty drafts (LLM declined: sources don't address the complaint) | 5 | n/a | |
| Decisions RESOLVE / REVIEW / ESCALATE | 76 / 15 / 9 | 81 / 18 / 1 | |
| **Confidence AUROC (predicting a good draft)** | **0.844** | 0.751 | |
| LLM-judge 1-5: relevance / completeness / specificity / correctness | 4.21 / 4.13 / 4.03 / 4.26 | n/a | directional only |
| End-to-end latency p50 / p95 | 6.4 s / 16.7 s | 1.0 s / 1.5 s | |
| Generation stage alone p50 / p95 | 0.93 s / 9.9 s | 0 | |
| Mean stage latency: generate / validate | 4.6 s / 3.9 s | 0 / 0.8 s | |

Quality by decision (LLM): **RESOLVE 0.832** reference-step recall (n = 76), **REVIEW 0.767** (n = 15),
**ESCALATE 0.194** (n = 9). The decision layer separates good drafts from bad ones much more sharply with the LLM
than with extractive drafts, because the LLM declines or writes thin drafts when the evidence is poor, and
validation and confidence pick that up. (The first 52 cases, reported earlier, gave the same picture: recall tied,
precision +0.065 with p = 0.024, AUROC 0.80; the full run tightens the estimates.)

**Interpretation.**
* On this dataset the LLM's value is **judgment and focus, not recall**: more precise drafts (fewer irrelevant
  steps), refusals when sources don't fit, and better-separated confidence. Recall is statistically tied with the
  extractive generator. Extractive is a deliberately strong baseline here because historical resolutions are
  templated; on messier real tickets the abstractive advantage should grow, but that is untested.
* Paraphrased LLM steps cost validation time (3.9 s vs 0.8 s mean): quoted steps skip NLI, paraphrases don't.
* **Latency: p95 16.7 s misses the plan's 15 s MVP target, as measured.** The tail is free-tier rate limiting:
  generation alone has p50 0.93 s, but rate-limit retry waits push its mean to 4.6 s (max 186 s). On a paid tier the
  expected end-to-end p95 is roughly retrieval + generation + validation ≈ 0.3 + 2-10 + 4 s. That is an estimate
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

**Remaining gap:** with Qwen, the novel-class set has 8 of 60 complaints so far (the daily cap was reached after
the 100 generation cases); the 60-case novel analysis above therefore uses gpt-oss-20b. Resume with
`python evaluation/generation_eval.py --generator llm --judge --judge-model openai/gpt-oss-120b --delay 2`.

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
concurrency 8 on this laptop *without* the LLM. With the free-tier LLM (§4.2) end-to-end p95 is 16.7 s at
concurrency 1, over the target, driven by rate-limit waits.

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

### 6.4 Draft quality ratings (AI-rated stand-in; human ratings still open)

**What this is and is not.** The plan's Tier-2 human evaluation uses `data/evaluation/human_eval_sheet.xlsx`
(built by `evaluation/build_human_eval_sheet.py`): 50 Qwen drafts drawn from the first 52 cases of §4.2 (seed 7,
both "declined" drafts included,
shuffled), each with the complaint, the 8 sources the generator saw (re-retrieved with the production retriever and
verified to reproduce the original run), the cited draft and the reference fix; system outputs and LLM-judge scores
sit on a hidden sheet. **No human has rated it yet.** The project owner asked for the sheet to be filled
automatically, so the ratings below were produced by **an AI rater (Claude Opus 5.5), not a human**, in a separate
copy, `data/evaluation/ai_eval_sheet.xlsx`, which states this on its first page. The rater read only the complaint,
sources, draft and reference fix (blind to the hidden System sheet) and wrote a reason for every row. The blank
human workbook is unchanged and remains the way to get real human judgements. Analysis:
`python evaluation/human_eval_analysis.py --sheet data/evaluation/ai_eval_sheet.xlsx --rater "..."`
→ `experiments/results/ai_rater_eval.json`.

| Measure (50 drafts, AI rater) | Result |
|---|---|
| Relevance / completeness / correctness (1-5, mean, 95% CI) | 4.10 [3.68, 4.48] / 4.14 [3.70, 4.52] / 4.00 [3.58, 4.36] |
| Safe to use as-is | 72% (36 of 50) |
| **Precision of system RESOLVE** (auto-resolved drafts the rater called safe) | **76%** (29 of 38) |
| Rater correctness by system decision: RESOLVE / REVIEW / ESCALATE | 4.21 (n = 38) / 4.00 (n = 8) / 2.00 (n = 4) |
| Heuristic confidence AUROC for "safe" | 0.81 |
| Spearman: rater correctness vs heuristic confidence | 0.54 (p = 0.0001) |
| Spearman: rater completeness vs automatic reference-step recall | 0.81 |
| Spearman: rater correctness vs LLM-judge correctness | 0.92 |
| Decision agreement with the system (3-way) / Cohen's kappa | 64% / 0.15 |

Decision confusion (rows = rater, columns = system):

| | system RESOLVE | system REVIEW | system ESCALATE |
|---|---:|---:|---:|
| rater RESOLVE | 29 | 6 | 1 |
| rater REVIEW | 3 | 0 | 0 |
| rater ESCALATE | 6 | 2 | 3 |

**Findings.**
* **The 9 unsafe auto-resolves are the "grounded but wrong" failure mode** (§4.1, §6.3), now with a measured rate:
  6 are drafts for the wrong scenario (broadband speed-upgrade steps for a phone that never reaches 5G, DNS steps
  for a factory-reset router, evening-congestion steps for a 100 Mbps bottleneck, failed-payment steps for a
  duplicate charge that would take *another* payment from an overdrawn customer) and 3 mix the right fix with
  steps from a neighbouring scenario. All 9 had confidence 0.81-0.89, the same range as good drafts, because they are
  well grounded in the sources they cite; the sources are the wrong ones.
* **Disagreement is mostly on the cautious side.** The system sent 8 drafts to REVIEW or ESCALATE that the rater
  would send as-is: 6 REVIEWs, all with confidence above the RESOLVE threshold, so a policy rule (predicted critical
  severity or an unrecognised intent; the stored run does not record which) made the call, plus 1 ESCALATE at low
  confidence. Kappa is low because the policy rules and the rater use different criteria, not because the rankings
  disagree (AUROC 0.81).
* **The automatic signals track the AI rater**: reference-step recall (0.81) and the LLM judge (0.92) rank drafts
  much like the rater does. This supports using them for regression testing, but it is AI agreeing with AI; it does
  not replace human validation.
* The 2 declined drafts were judged correct declines (no retrieved source covered a new connection awaiting
  activation).

**Caveats.** An AI rater may share blind spots with the LLM judge and the generator; the generator (Qwen) and judge
(gpt-oss) are different model families from the rater, which limits self-preference but not shared biases. Treat
these as provisional until a human rates the same 50 rows (the analysis then reports human-vs-AI agreement).

### 6.5 Outage drill: database stopped under a running API

Graceful degradation (ARCHITECTURE.md §6.1) checked on the real stack, not only with unit-test fakes: the database
container was stopped while the API kept serving, then restarted (`experiments/results/outage_drill.txt`, procedure
included).

| Phase | Response | Client time |
|---|---|---:|
| Healthy | 200, RESOLVE, 4 cited steps | 8.4 s |
| Database stopped, requests 1-2 (circuit closed) | 200, ESCALATE "Knowledge base unavailable", flags `retrieval_unavailable`, `understanding_unavailable` | 13.5 s, 10.0 s |
| Request 3 (circuit opens after 5 database failures) | 200, ESCALATE | 3.4 s |
| Requests 4-5 (circuit open) | 200, ESCALATE, database not touched | 0.05 s |
| `/health` during the outage | `degraded`, `database_circuit: open` | |
| Database back, circuit still open | 200, ESCALATE | 0.10 s |
| After the 30 s recovery window | 200, RESOLVE, 4 cited steps; `/health` back to `healthy`, circuit `closed` | 7.1 s |

No request failed, nothing was resolved without evidence, and the service recovered by itself. The first requests
of an outage are slow because each waits up to `DB_QUERY_TIMEOUT_S` (10 s) before the circuit opens; a lower timeout
trades that wait for more false alarms under load.

## 7. Engineering checks

| Check | Result |
|---|---|
| Unit + API + DB integration tests (`pytest`) | 91 passed (DB tests run against the live pgvector container; they skip if no DB) |
| Line coverage of `app/` | 88% |
| `black --check`, `flake8`, `mypy` (app, scripts, evaluation, experiments, tests: 75 files) | clean |
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
* LLM generation evaluated on all 100 cases with Qwen, 19 with Gemini, and the 60 novel-class cases with
  gpt-oss-20b (Qwen has 8 of them): free-tier daily caps (200k tokens/day on Groq, 20 requests/day on Gemini)
  prevented one model from covering everything.
* E3 test half is small (100 items); the TVD drift coefficient was calibrated on this dataset's class mix.
* One worker sustains about 1 request/s on a laptop CPU (§6.1); horizontal scaling or GPU inference is needed for volume.
* Heuristic confidence is uncalibrated (AUROC 0.75 extractive / 0.80 LLM for predicting a good draft); the
  feedback endpoint collects the data needed to calibrate it.

## 9. Future work (prioritised by expected impact on the measured weaknesses)

1. **Per-request "wrong scenario" detection** (the "grounded but wrong" mode, §4.1 and §6.3). Add (a) semantic vs
   lexical *ranker disagreement* as an uncertainty signal in the confidence score, and (b) a complaint ↔ cited-source
   relevance check with the cross-encoder applied to the 3–5 cited sources only. Evaluate on the 60 novel-class
   complaints and the §6.3 example; success = fewer confident wrong RESOLVEs with no loss of RESOLVE rate in-distribution.
2. **Sentence-level / late-interaction retrieval** so boilerplate sentences ("I work from home…") cannot outvote the
   symptom sentence; re-run E1.
3. **Calibrate the confidence score** from the human ratings (§6.4) and the feedback endpoint (isotonic or Platt on
   "safe to use"); then run Experiment 5 to set RESOLVE/REVIEW thresholds for a target precision.
4. **Intent:** evaluate the LLM classifier (`INTENT_CLASSIFIER=llm`) against k-NN (target macro-F1 > 0.75), and
   consider a hybrid (k-NN unless the vote is split).
5. **Throughput:** batched GPU inference for NLI and embeddings, then multiple workers; re-run the load test (target
   > 5 req/s) and re-run E2 on GPU to decide whether reranking (+0.052 P@5) becomes affordable.
6. **Complete the novel-class LLM run** with the deployed model (Qwen has 8 of 60; the 100 generation cases are done).
7. **Real data:** re-run every evaluation on real (anonymised) tickets; the synthetic set fixes relevance labels but
   not linguistic variety.
8. **Switch to all-mpnet-base-v2** (E4: +0.16 nDCG@10) with a full re-baseline: 768-d columns, re-embed,
   re-tune E3 thresholds and sufficiency floors on the new cosine scale, re-run all evaluations.
   Optional experiments not run: E6 LLM temperature, E7 context length (free-tier LLM quota), E8 caching.
9. **Out-of-domain stress test** with the Hugging Face tickets (§1): measure how often the system RESOLVEs IT
   tickets it has no knowledge for, and whether the drift monitor alerts.

