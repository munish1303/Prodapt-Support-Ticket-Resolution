# Evaluation

All numbers are copied from files in `experiments/results/` (and `data/evaluation/leakage_report.json`)
produced by the scripts named in each section. Where a run is incomplete (the LLM evaluation, capped by the
free-tier daily token quota) the coverage is stated explicitly. Nothing is estimated.

**Embedding model.** The system uses `all-mpnet-base-v2` for retrieval and understanding since Experiment 4 (it
replaced `all-MiniLM-L6-v2`; groundedness validation keeps MiniLM, see EXPERIMENTS.md E3/E4). All numbers below are
from the re-baselined mpnet system unless marked otherwise; the MiniLM results are kept for comparison in
`experiments/results/minilm_baseline/` and quoted as "before".

## Plan targets vs measured

The plan's provisional targets (IMPLEMENTATION_PLAN.md §4.2 / §40.2) against what was measured. Missed targets are
reported as missed; the sections below explain each number.

| Target | Measured | Met | Where |
|---|---|:---:|---|
| Intent classification macro-F1 > 0.75 | **0.889** (500 held-out complaints; 0.723 before the embedding switch) | ✓ | §2.1 |
| Product extraction F1 > 0.70 | 0.814 (0.716 before) | ✓ | §2.1 |
| Hybrid retrieval Recall@10 > 0.70 | **0.726** (capped Recall@10, E1 test split; plain Recall@10 is uninformative when a query has dozens of equally relevant tickets, §1.1); 0.558 before the embedding switch | ✓ | EXPERIMENTS.md E1, E4 |
| Citation accuracy > 0.90 | 1.000 (extractive); LLM **0.990** (1.000 on the 99 non-empty drafts; 0.950 before the switch) | ✓ | §4 |
| Groundedness > 0.80 | 1.000 (extractive); LLM **0.988** (0.942 before); E3 multi-method checker F1 0.918 | ✓ | §4, §5 |
| P95 latency < 15 s (MVP) | 1.6 s without LLM (concurrency 1); with the free-tier LLM **14.7 s** on the 67 cases run under the current rate-limit settings (16.7 s before the switch) | ✓ (just, free tier) | §4.2, §6.1 |
| Throughput > 5 req/s | 1.0 req/s at concurrency 1, 1.34 req/s at concurrency 8 per worker on the laptop CPU, 0% errors | ✗ | §6.1 |
| Test coverage > 70% | 88% of `app/` | ✓ | §7 |
| Experiments 1-3 run, with decisions | E1, E2, E3 done; optional E4 (embeddings) and E5 (thresholds, on AI ratings) done; E4′ evolving classes | ✓ | EXPERIMENTS.md |

The embedding switch (EXPERIMENTS.md E4) moved intent F1 and Recall@10 above their targets. The remaining miss,
throughput, comes from CPU inference in one worker; LLM latency meets the target only just, on the free tier. The
levers are more workers, batched GPU inference and a paid LLM tier (§9).

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

| Field | Metric | mpnet | before (MiniLM) | Plan target (provisional) |
|---|---|---:|---:|---:|
| Intent | accuracy | **0.890** | 0.724 | |
| | macro-F1 | **0.889** | 0.723 | > 0.75 (met) |
| | flagged unknown (false-unknown rate) | 0.010 | 0.026 | |
| | accuracy when not flagged unknown | 0.899 | 0.743 | |
| Products | micro precision / recall / F1 | 0.770 / 0.863 / **0.814** | 0.689 / 0.745 / 0.716 | > 0.70 (met) |
| | exact set match | 0.660 | 0.558 | |
| | primary product recalled | 0.898 | 0.786 | |
| Severity (4 levels) | accuracy | **0.884** | 0.784 | |
| | macro-F1 | 0.876 | 0.768 | |
| | within one level | 0.970 | 0.930 | |
| Sentiment (3 classes) | accuracy | **0.790** | 0.790 | |
| | macro-F1 | 0.742 | 0.742 | |
| Latency | understanding per complaint (CPU, in-memory index) | mean 409 ms, p95 553 ms | mean 266 ms, p95 319 ms | |

`python evaluation/understanding_eval.py --index memory` (exact k-NN over the 2,043 corpus complaints; the same
mode as the MiniLM run, so the comparison isolates the embedding model). Sentiment uses its own model and is
unaffected. Latency rises because mpnet is a larger model (the mpnet run was also inside a 2-core container).

Per-class intent F1 (mpnet; before in brackets): technical_support 0.990 (0.855), billing_dispute 0.939 (0.809),
outage_report 0.925 (0.830), speed_issue 0.917 (0.746), account_management 0.886 (0.667), connectivity_issue 0.879
(0.661), hardware_problem 0.863 (0.636), service_activation 0.832 (0.781), **plan_change 0.769** (0.528). Every
class improved; the weakest are still the semantically overlapping ones (plan changes vs billing, both about price).

Severity confusion (rows = true, cols = predicted; low/medium/high/critical):
`[[115, 4, 0, 0], [2, 159, 8, 15], [0, 1, 80, 26], [0, 0, 2, 88]]`. The main error is still high → critical (26;
31 before), where urgency cues ("whole street", "no internet at all") override the neighbour-based base level.

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

| min vote share | min similarity | dev false-unknown | dev accuracy | novel complaints flagged | before (MiniLM): false-unknown / accuracy / novel |
|---:|---:|---:|---:|---:|---|
| 0.00 | 0.00 | 0.000 | 0.889 | 0.000 | 0.000 / 0.729 / 0.000 |
| **0.35** | **0.40** | **0.005** | **0.889** | **0.017** | 0.025 / 0.719 / 0.050 |
| 0.45 | 0.40 | 0.030 | 0.874 | 0.033 | 0.091 / 0.688 / 0.133 |
| 0.55 | 0.40 | 0.111 | 0.839 | 0.233 | 0.206 / 0.628 / 0.483 |
| 0.55 | 0.55 | 0.141 | 0.809 | 0.233 | 0.271 / 0.578 / 0.600 |

Selection rule (unchanged): the highest dev accuracy among settings that flag some novel complaints with at most
3% false unknowns. It selects 0.35 / 0.40 again, so the thresholds were kept after the switch.

**Finding: k-NN vote share is a weak novelty detector**, and better embeddings made it weaker. Roaming complaints
sit close to real connectivity and billing tickets, so the vote is confident but wrong; with mpnet they look even
more familiar (catching a quarter of them would flag 11% of normal traffic). The chosen operating point favours accuracy on known classes; novel classes are meant to be
caught downstream (low relevance/groundedness ⇒ ESCALATE; measured in §4 when run) and by the drift monitor.

**Evolving classes, offline** (60 held-out roaming complaints):

| | before the class exists | after registering `roaming_issue` + ingesting 221 wave-2 tickets |
|---|---|---|
| predicted `roaming_issue` | n/a | **88.3%** (53/60); 85.0% before the switch |
| flagged unknown | 1.7% (1/60); 5.0% before | 1.7% |
| other predictions | billing 20, outage 20, connectivity 12, service activation 7 | connectivity 5, outage 1 |
| in-distribution intent accuracy (500 set) | 0.890 | **0.888** (0.724 → 0.682 before the switch) |

No retraining or redeploy: the class becomes predictable from ingestion alone. **Cost:** with mpnet, in-distribution
accuracy barely moves (0.890 → 0.888); with MiniLM it dropped by 0.042 because roaming tickets attracted mobile
complaints of other classes. In production the new
class's tickets should be reviewed before ingestion, and the per-class F1 rechecked after.

## 3. Retrieval (L1) and baseline comparison (L3)

**Script:** `python experiments/e1_retrieval.py` (see EXPERIMENTS.md E1).
Metrics: nDCG@10 (graded), MRR@20, P@5, Hit@5, capped Recall@10 (`|rel ∩ top10| / min(10, |rel|)`, because many
queries have dozens of equally relevant tickets, which makes plain Recall@10 uninformative), KB-Recall@10.

**Results** (200 held-out queries; full table, CIs and significance tests in EXPERIMENTS.md E1):

| Configuration | nDCG@10 | MRR | P@5 | Hit@5 | Recall@10 (capped) |
|---|---:|---:|---:|---:|---:|
| keyword baseline (today's practice) | 0.221 | 0.270 | 0.208 | 0.405 | 0.197 |
| lexical, `plainto_tsquery` (AND) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| lexical, OR-of-terms FTS | 0.274 | 0.452 | 0.270 | 0.565 | 0.242 |
| semantic (pgvector, mpnet) | 0.716 | 0.752 | 0.704 | 0.870 | 0.721 |
| **hybrid RRF 0.9:0.1 (weight chosen on dev)** | **0.729** | **0.793** | **0.722** | **0.890** | **0.726** |
| hybrid + cross-encoder rerank (E2; disabled: +2 s) | 0.736 | 0.803 | 0.732 | 0.895 | |
| *before the embedding switch:* semantic (MiniLM) | 0.557 | 0.600 | 0.545 | 0.715 | 0.559 |
| *before:* hybrid RRF 0.8:0.2 (MiniLM) | 0.572 | 0.657 | 0.561 | 0.760 | 0.558 |

**Deployed configuration** (`python evaluation/retrieval_eval.py` → `experiments/results/retrieval_production.json`):
the generator receives **8 sources** (`RETRIEVAL_TOP_K`) with 2 guaranteed KB slots. On the same 200 queries: MRR 0.794,
P@5 0.722, Hit@5 0.890 (before: 0.657 / 0.561 / 0.760), and **the scenario's KB article is in the generator's context
for 87% of complaints (77% before), vs 14% without KB slots** (E1 hybrid, top 10). The slots trade two ticket
positions for the canonical procedure. nDCG@10 (0.593) and P@10 are lower than in E1 only because the list is 8 long
with 2 KB positions. Latency p50 282 ms / p95 449 ms (2-core container).

**Baseline comparison (L3):** keyword search finds a highly relevant ticket in the top 5 for 40.5% of complaints;
the deployed hybrid retriever does for 89.0% (+0.485, p < 0.001; 76.0% before the embedding switch). Hybrid's gain
over semantic-only is significant but small (nDCG@10 +0.014, p = 0.0005; MRR +0.041, p = 0.003).

## 4. Generation and pipeline (L2)

**Script:** `python evaluation/generation_eval.py --generator extractive|llm` on 100 held-out complaints.
Metrics: citation accuracy, citation coverage, groundedness (validation service), reference-step recall and
step precision (generated vs scenario fix steps, MiniLM cosine ≥ 0.6), scenario KB article retrieved, decision
mix, latency per stage, **AUROC of heuristic confidence for predicting a good draft** (reference-step recall ≥ 0.5),
and the decision mix on 60 novel-class complaints. Optional LLM-as-judge (directional only: LLM judges favour
LLM text and are not a substitute for human review).

### 4.1 Extractive generator (no LLM) ✅

`experiments/results/generation_extractive.json` (100 held-out complaints + 60 novel-class complaints).

| Metric | mpnet | before (MiniLM) |
|---|---:|---:|
| Citation accuracy / coverage | 1.000 / 1.000 | 1.000 / 1.000 |
| Groundedness | 1.000 (steps are quoted verbatim from sources) | 1.000 |
| Reference-step recall / step precision | **0.829** / **0.624** | 0.743 / 0.560 |
| Steps per draft | 5.0 | 5.0 |
| Scenario KB article among sources | 0.86 | 0.78 |
| Intent correct (pipeline, pgvector ANN) | 0.88 | 0.69 |
| Mean heuristic confidence | 0.899 | 0.871 |
| Decisions (RESOLVE / REVIEW / ESCALATE) | 86 / 14 / 0 | 81 / 18 / 1 |
| Confidence AUROC for predicting a good draft (step recall ≥ 0.5) | **0.891** | 0.751 |
| End-to-end latency p50 / p95 / max | 908 / 1,585 / 2,531 ms | 1,014 / 1,464 / 4,109 ms |
| Mean stage latency: embed / understand+retrieve / validate | 195 / 245 / 515 ms | 47 / 272 / 769 ms |
| **Novel-class complaints (60): RESOLVE / REVIEW / ESCALATE** | **31 / 29 / 0** | 29 / 30 / 1 |

Quality by decision: RESOLVE drafts recover 0.824 of reference steps, REVIEW 0.857 (REVIEW here comes from policy
rules such as critical severity, not from weak drafts). Confidence now separates good from weak drafts much better
(AUROC 0.89 vs 0.75), because better retrieval makes the relevance signals more informative.

*Latency note.* With mpnet doing validation's sentence embeddings, validation took 4.9 s per request (mpnet is 6x
slower than MiniLM on ~80 source sentences); a cross-request cache cut that to 3.0 s. Keeping MiniLM for validation
(EXPERIMENTS.md E3: same quality) brought it to 0.5 s, so end-to-end latency stayed at MiniLM's level. All three runs
produced identical quality numbers.

**Finding: "grounded but wrong".** Half of the complaints about a ticket class the system has never seen
(roaming) are RESOLVEd (31 of 60 with mpnet, 29 with MiniLM: better embeddings make unseen complaints look more
familiar, not less). The drafts are faithfully grounded, just in the wrong tickets: roaming complaints are
similar enough to mobile-connectivity tickets (cosine ≈ 0.7) to pass the relevance gate, and the extractive
generator has no notion of "these sources don't answer this question". Groundedness verifies *draft ↔ sources*,
not *sources ↔ complaint*. Mitigations: the LLM generator is instructed to return no steps when sources don't
address the complaint (measured in §4.2: on 47 unseen complaints, wrong RESOLVEs fall from 51% with extractive
drafts to 30% with Qwen); a complaint↔source relevance verifier (e.g. the cross-encoder from E2
applied to the 3–5 cited sources only) is the natural next safeguard; drift monitoring watches the
nearest-neighbour similarity of incoming traffic.

**First-run note.** The first extractive run (before E3 v2) scored groundedness 0.571 and escalated 88/100 drafts
for "contradictions" on verbatim-quoted steps, which exposed the NLI premise issue fixed in E3 v2 (EXPERIMENTS.md).

### 4.2 LLM generator

`experiments/results/generation_llm.json`. Generator `qwen/qwen3.8-27b`, judge `openai/gpt-oss-120b`
(different model family, to reduce self-preference bias), Groq free tier. The run before the embedding switch is kept
in `experiments/results/minilm_baseline/generation_llm.json`: same 100 complaints, same generator, prompt, judge and
metric embedder, so the "before" column isolates the effect of better retrieval and understanding.

**Coverage: all 100 planned cases** (GEN-0001 to GEN-0100, the fixed pre-shuffled eval order). Groq's free tier caps
each model at **200,000 tokens per day** (about 110 drafts), so the run took two sessions: 33 cases on 3 October, 67 on
the night of 4-5 October. Rate-limited attempts (7) were retried later rather than scored as failures, so no case was
scored in a degraded state. The extractive generator ran on the same 100 complaints, so every comparison is paired.

| Metric (n = 100, paired) | LLM, mpnet | LLM, before (MiniLM) | Extractive, mpnet |
|---|---:|---:|---:|
| Reference-step recall | **0.865** [95% CI 0.802-0.921] | 0.765 | 0.828 |
| Step precision | **0.736** [0.677-0.785] | 0.626 | 0.624 |
| Groundedness | 0.988 (0.998 over the 99 non-empty drafts) | 0.942 | 1.000 |
| Citation accuracy / coverage | 0.990 / 0.990 (1.000 over the 99 non-empty drafts) | 0.950 / 0.950 | 1.000 / 1.000 |
| Scenario KB article among sources | 0.86 | 0.78 | 0.86 |
| Intent correct | 0.88 | 0.69 | 0.88 |
| Steps per draft | 4.52 | 4.54 | 5.00 |
| Drafts with a contradicted step | 0 | 3 | 0 |
| Empty drafts (LLM declined: sources don't address the complaint) | 1 | 5 | n/a |
| Decisions RESOLVE / REVIEW / ESCALATE | 85 / 14 / 1 | 76 / 15 / 9 | 86 / 14 / 0 |
| Confidence AUROC (predicting a good draft) | 0.850 | 0.844 | 0.891 |
| LLM-judge 1-5: relevance / completeness / specificity / correctness | 4.60 / 4.53 / 4.30 / 4.63 | 4.21 / 4.13 / 4.03 / 4.26 | n/a |
| End-to-end latency p50 / p95 | 8.1 s / 14.7 s (see latency note) | 6.4 s / 16.7 s | 0.9 s / 1.6 s |
| Generation stage alone p50 / p95 | 3.7 s / 11.1 s | 0.93 s / 9.9 s | 0 |
| Mean validation time | 3.4 s | 3.9 s | 0.5 s |

Paired bootstrap, LLM mpnet vs LLM before: reference-step recall **+0.100 (p = 0.003)**, step precision **+0.110
(p < 0.001)**, groundedness +0.046 (p = 0.001), citation accuracy +0.040 (p = 0.019). LLM vs extractive, both mpnet:
step precision **+0.112 (p < 0.001)**, reference-step recall +0.037 (p = 0.08, not significant).

Quality by decision (LLM, mpnet): RESOLVE 0.870 reference-step recall (n = 85), REVIEW 0.893 (n = 14, sent there by
policy rules such as critical severity, not weak drafts), ESCALATE 0.0 (n = 1, the one declined draft). Before the
switch: RESOLVE 0.832 (76), REVIEW 0.767 (15), ESCALATE 0.194 (9).

**Interpretation.**
* **Better sources helped the LLM more than the extractive generator**: +0.100 recall and +0.110 precision for the LLM,
  against +0.086 and +0.064 for extractive (§4.1). The LLM's value on this dataset is still **focus, not recall**: it
  writes more precise drafts than extractive (+0.112 precision) while recall is statistically tied. Extractive is a
  deliberately strong baseline because historical resolutions are templated; on messier real tickets the abstractive
  advantage should grow, but that is untested.
* **Declines and contradictions almost disappeared** (5 → 1 empty drafts, 3 → 0 contradicted drafts): with the right
  KB article in context more often, the LLM rarely has reason to decline. The flip side is that the ESCALATE bucket is
  nearly empty: the bad drafts that remain are wrong-scenario drafts that look confident, which the AI ratings in §6.4
  expose and which confidence cannot catch.
* **Latency: p95 14.7 s, just under the plan's 15 s MVP target, on the free tier.** Measured on the 67 cases run under
  the current rate-limit settings (a request waits at most 10 s on a per-minute limit, then falls back to the
  extractive draft; ARCHITECTURE.md §6.1). The 33 first-session cases ran before that setting existed and six of them
  waited 1-3 minutes on per-minute limits; over all 100 cases p50 is 8.3 s and p95 63 s. Generation p50 is 3.7 s
  against 0.93 s in the earlier run: Groq's 8,000 tokens/minute allows about 4.5 drafts a minute and the evaluation ran
  faster than that, so part of the generation time is likely waiting. The run's logs were not kept, so that split is
  not measured. On a paid tier the expected p95 is roughly retrieval + generation + validation ≈ 0.7 + 2-10 + 3.4 s
  (an estimate from the measured stages).
* Paraphrased LLM steps cost validation time (3.4 s vs 0.5 s mean): quoted steps skip NLI, paraphrases don't.
* LLM-judge scores are directional only (single judge, no human calibration). The bootstrap covers case-to-case
  variation, not sampling variation of the generator (temperature 0.2).

**Second LLM family (cross-check): Gemini** (measured before the embedding switch; all three columns below come
from the MiniLM system, so compare them with each other, not with the table above).
`experiments/results/generation_llm_gemini.json`, generator
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

**Novel-class complaints with an LLM generator: the "grounded but wrong" test.** The same 60 roaming complaints as
§4.1 (a class absent from the corpus). With the deployed Qwen model on the mpnet system, **47 of 60** were scored
before the daily quota ran out; the other 13 hit the quota and are not counted (`generation_llm.json`,
`novel_rows`). An earlier run with Groq `openai/gpt-oss-20b` (low reasoning effort, MiniLM system, before the switch)
covered all 60 (`experiments/results/generation_llm_gptoss20b_novel.json`). On the 47 complaints all three share:

| Unseen-issue complaints (n = 47, paired) | RESOLVE (confident wrong answer) | REVIEW | ESCALATE | Drafts declined (no steps) |
|---|---:|---:|---:|---:|
| Extractive generator, mpnet (§4.1) | **24 (51%)** | 23 | 0 | n/a |
| LLM Qwen 3.8 27B, mpnet (deployed) | **14 (30%)** | 19 | 14 | 13 (28%) |
| LLM gpt-oss-20b, MiniLM (before the switch) | 12 (26%) | 12 | 23 | 19 (40%) |

On all 60 complaints: extractive mpnet 31 RESOLVEs (52%), gpt-oss-20b 13 (22%).

The prompt rule *"if the sources do not address the complaint, return an empty list"* does real work: Qwen declined
13 of 47, which the pipeline turns into ESCALATE ("insufficient evidence"), and confident wrong answers on an unseen
issue type fall from 51% to 30%. It does not eliminate them, so the traffic-level intent-mix drift alert (E4′)
remains the second line of defence. Qwen vs gpt-oss-20b is not a clean model comparison (different retrieval
systems); both show the same effect.

**Remaining gap:** 13 of the 60 unseen-issue cases with Qwen (daily quota). Resume with
`python evaluation/generation_eval.py --generator llm --judge --judge-model openai/gpt-oss-120b --delay 2`
(only missing cases run).

## 5. Groundedness validation (L1)

See EXPERIMENTS.md E3. Held-out F1 for detecting not-supported claims (v2, multi-method): **0.918**
(accuracy 0.930) vs 0.897 NLI only, 0.739 lexical only and 0.694 similarity only. False alarms: 0% on verbatim-quoted
steps and 15% on paraphrased steps. The first version (chunk premises, max contradiction) scored 0.736 on the same
set, with 40% false alarms on verbatim steps; the generation eval exposed this and drove the v2 design.
After the embedding switch E3 was re-run with both embedding models on a widened grid: 0.918 with MiniLM (deployed
for validation, thresholds unchanged) vs 0.927 with mpnet, a difference within noise at 6x the cost (E3, E4).

## 6. System health (L4)

### 6.1 Load test

**Script:** `python evaluation/system_eval.py --url http://localhost:8000 --tag extractive` (40 requests per level,
concurrency 1, 4, 8) → `experiments/results/system_load_extractive.json`. One uvicorn worker in the API container on
the laptop CPU (no GPU), API run with `LLM_PROVIDER=extractive` so the free-tier LLM quota does not dominate; LLM
generation latency is reported separately in §4.2. Distinct held-out complaints.

| Concurrency | Throughput | Client latency p50 / p95 / p99 | Server latency p50 / p95 | Error rate | before (MiniLM): throughput, p50 |
|---:|---:|---|---|---:|---|
| 1 | 0.96 req/s | 1,009 / 1,541 / 1,746 ms | 985 / 1,512 ms | 0% | 0.96 req/s, 1,023 ms |
| 4 | 1.20 req/s | 3,172 / 4,426 / 5,269 ms | 3,146 / 4,342 ms | 0% | 0.83 req/s, 4,614 ms |
| 8 | 1.34 req/s | 5,200 / 9,205 / 10,567 ms | 5,152 / 9,166 ms | 0% | 0.84 req/s, 9,253 ms |

**Reading.** Zero errors under load. Throughput now rises a little with concurrency (1.34 vs 0.84 req/s at 8) because
validation reuses cached sentence embeddings of sources that recur across requests; inference is still CPU-bound and
serialised in one worker, so latency grows with concurrency. Server latency ≈ client latency: the queue is inside the
service. Scaling path, in order: more workers or pods behind a load balancer (a process with all models measured
~1.2 GB resident), then a batched inference service (GPU) for NLI and embeddings. The plan's MVP target (P95 < 15 s)
holds up to concurrency 8 *without* the LLM. With the free-tier LLM (§4.2) end-to-end p95 is 14.7 s at concurrency
1, just under the target (16.7 s before the switch, when long rate-limit waits were still allowed).

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
(built by `evaluation/build_human_eval_sheet.py --case-limit 52`): 50 Qwen drafts drawn from the first 52 cases of
§4.2 (seed 7, any "declined" drafts included, shuffled), each with the complaint, the 8 sources the generator saw
(re-retrieved with the production retriever and verified to reproduce the run for all 50), the cited draft and the
reference fix; system outputs and LLM-judge scores sit on a hidden sheet. **No human has rated it yet.** The project
owner asked for the sheet to be filled automatically, so the ratings below were produced by **an AI rater (Claude
Opus 5.5), not a human**, in a separate copy, `data/evaluation/ai_eval_sheet.xlsx`, which states this on its first
page. The rater read only the complaint, sources, draft and reference fix (blind to the hidden System sheet) and wrote
a reason for every row. The blank human workbook remains the way to get real human judgements. Analysis:
`python evaluation/human_eval_analysis.py --sheet data/evaluation/ai_eval_sheet.xlsx --rater "..."`
→ `experiments/results/ai_rater_eval.json`.

The sheet was rebuilt from the mpnet run. The pre-switch sheet, its AI-rated copy and analysis are kept in
`experiments/results/minilm_baseline/` (`*_minilm_drafts.xlsx`, `ai_rater_eval.json`). The same seed picks 48 of
the same 50 cases, so the two ratings can be compared case by case.

| Measure (50 drafts, AI rater) | mpnet | before (MiniLM) |
|---|---|---|
| Relevance / completeness / correctness (1-5, mean, 95% CI) | **4.58** [4.26, 4.84] / **4.60** [4.26, 4.86] / **4.52** [4.22, 4.78] | 4.10 / 4.14 / 4.00 |
| Safe to use as-is | **84%** (42 of 50) | 72% (36 of 50) |
| Precision of system RESOLVE (auto-resolved drafts the rater called safe) | **81%** (35 of 43) | 76% (29 of 38) |
| Rater correctness by system decision: RESOLVE / REVIEW / ESCALATE | 4.44 (n = 43) / 5.00 (n = 7) / none | 4.21 (38) / 4.00 (8) / 2.00 (4) |
| Heuristic confidence AUROC for "safe" | **0.63** [0.48, 0.79] | 0.81 |
| Spearman: rater correctness vs heuristic confidence | 0.10 (p = 0.50) | 0.54 (p = 0.0001) |
| Spearman: rater completeness vs automatic reference-step recall | 0.73 | 0.81 |
| Spearman: rater correctness vs LLM-judge correctness | 0.88 | 0.92 |
| Decision agreement with the system (3-way) / Cohen's kappa | 70% / -0.13 | 64% / 0.15 |

Decision confusion, mpnet (rows = rater, columns = system):

| | system RESOLVE | system REVIEW | system ESCALATE |
|---|---:|---:|---:|
| rater RESOLVE | 35 | 7 | 0 |
| rater REVIEW | 4 | 0 | 0 |
| rater ESCALATE | 4 | 0 | 0 |

**Findings.**
* **Drafts got better.** On the 48 shared cases, 40 drafts are safe to use as-is against 35 before: 6 fixed (among
  them the duplicate-charge draft that would have taken another payment from an overdrawn customer, the village-wide
  mast outage, and two new fibre connections), 1 got worse (overnight planned maintenance, now mixed with Wi-Fi
  interference steps), 7 still wrong.
* **Every remaining unsafe draft was auto-resolved** (8 of 8), with confidence 0.83-0.91. Four are drafts for the wrong
  scenario (broadband speed-upgrade steps for a phone with a 5G problem, twice; steps for a self-rebooting router
  after a factory reset; factory-reset steps for a new fibre install that was never provisioned) and four mix the
  right fix with steps from a neighbouring scenario. All are well grounded in the sources they cite; the sources are
  for a confusable problem.
* **So confidence no longer ranks these drafts** (AUROC 0.63, CI down to chance, against 0.81 before). Better
  retrieval removed the *easy* failures, where similarity was low and confidence low with it; what remains are
  confusable-scenario errors that look exactly like good drafts to every signal the confidence score uses. On the
  100-case automatic evaluation confidence still separates good from bad drafts (AUROC 0.85, §4.2) because the
  label there is reference-step recall over a larger set. Both are true, and the first is the one that matters
  for auto-resolving. This makes per-request wrong-scenario detection (§9, item 1) the most valuable next step.
* **Disagreement is mostly on the cautious side otherwise.** The 7 system REVIEWs were all rated safe (correctness
  5.0); policy rules (predicted critical severity or an unrecognised intent) sent them to review. Kappa is negative
  because the system never ESCALATEs in this sample while the rater would escalate 4, and REVIEW is assigned by
  different criteria.
* **The automatic signals still track the AI rater**: reference-step recall (0.73) and the LLM judge (0.88). This
  supports using them for regression testing, but it is AI agreeing with AI; it does not replace human validation.

**Caveats.** An AI rater may share blind spots with the LLM judge and the generator; the generator (Qwen) and judge
(gpt-oss) are different model families from the rater, which limits self-preference but not shared biases. n = 50
with 8 unsafe drafts, so the AUROC and precision estimates are wide. Treat these as provisional until a human rates
the same 50 rows (the analysis then reports human-vs-AI agreement).

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
| Unit + API + DB integration tests (`pytest`) | 102 passed (DB tests run against the live pgvector container; they skip if no DB) |
| Line coverage of `app/` | 88% |
| `black --check`, `flake8`, `mypy` (app, scripts, evaluation, experiments, tests: 79 files) | clean |
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
* Intent macro-F1 is 0.889 after the embedding switch (0.723 before); the LLM classifier (`INTENT_CLASSIFIER=llm`)
  has not been evaluated.
* Better embeddings made unseen issue types look more familiar (novel-class RESOLVEs 29 → 31 of 60, extractive), so
  the per-request novelty signal got weaker; the traffic-level intent-mix alert still fires (E4′).
* LLM generation evaluated on all 100 cases with Qwen (before and after the switch), 19 with Gemini and the 60
  novel-class cases with gpt-oss-20b (before the switch); Qwen covers 47 of the 60 novel-class cases. Free-tier
  daily caps (200k tokens/day on Groq, 20 requests/day on Gemini) prevented one model from covering everything.
* E3 test half is small (100 items); the TVD drift coefficient was calibrated on this dataset's class mix.
* One worker sustains about 1-1.3 requests/s on a laptop CPU (§6.1); horizontal scaling or GPU inference is needed for volume.
* Heuristic confidence is uncalibrated. It separates good from bad drafts on the automatic labels (AUROC 0.89
  extractive, 0.85 LLM) but not the AI rater's "safe" on 50 drafts (0.63): after the switch the unsafe drafts are
  confusable-scenario drafts with confidence 0.83-0.91 (§6.4). The feedback endpoint collects the data needed to
  calibrate it.

## 9. Future work (prioritised by expected impact on the measured weaknesses)

1. **Per-request "wrong scenario" detection** (the "grounded but wrong" mode, §4.1 and §6.3). Add (a) semantic vs
   lexical *ranker disagreement* as an uncertainty signal in the confidence score, and (b) a complaint ↔ cited-source
   relevance check with the cross-encoder applied to the 3–5 cited sources only. Evaluate on the 60 novel-class
   complaints, the §6.3 example and the 8 unsafe auto-resolves of §6.4 (all confusable scenarios, confidence
   0.83-0.91); success = fewer confident wrong RESOLVEs with no loss of RESOLVE rate in-distribution.
2. **Sentence-level / late-interaction retrieval** so boilerplate sentences ("I work from home…") cannot outvote the
   symptom sentence; re-run E1.
3. **Calibrate the confidence score** from the human ratings (§6.4) and the feedback endpoint (isotonic or Platt on
   "safe to use"); then run Experiment 5 to set RESOLVE/REVIEW thresholds for a target precision.
4. **Intent:** evaluate the LLM classifier (`INTENT_CLASSIFIER=llm`) against k-NN (now 0.889 macro-F1), mainly for
   the weakest class (plan_change 0.77) and for novelty detection.
5. **Throughput:** batched GPU inference for NLI and embeddings, then multiple workers; re-run the load test (target
   > 5 req/s). Reranking is no longer worth it after the switch (+0.010 P@5, E2).
6. **Complete the novel-class LLM run** with the deployed model (Qwen has 47 of 60; the 100 generation cases are done).
7. **Real data:** re-run every evaluation on real (anonymised) tickets; the synthetic set fixes relevance labels but
   not linguistic variety.
8. **Precompute source sentence embeddings** at ingestion (today a cross-request cache) so validation never embeds
   sources at request time; then mpnet could also be used for validation at no latency cost.
   Optional experiments not run: E6 LLM temperature, E7 context length (free-tier LLM quota), E8 caching.
9. **Out-of-domain stress test** with the Hugging Face tickets (§1): measure how often the system RESOLVEs IT
   tickets it has no knowledge for, and whether the drift monitor alerts.

