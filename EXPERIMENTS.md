# Experiments

Every number below is copied from a results file in `experiments/results/`, produced by the script named in
the section. Where an experiment was iterated (E1, E3, E4′), the earlier run is kept and the reason for the change
is documented.

| # | Question | Status | Decision |
|---|---|---|---|
| E1 | Does hybrid retrieval beat semantic-only, lexical-only and keyword search? | **Done** | Hybrid RRF 0.8:0.2 (weight chosen on dev) |
| E2 | Does cross-encoder reranking pay for its latency? | **Done** | +0.052 P@5 but +2.7 s on CPU, so kept **off** |
| E3 | Does multi-method groundedness beat single signals? | **Done** (v2) | Multi-method v2: sentence premises + quote rule; F1 0.918 |
| E4′ | Can the system absorb a new ticket class without retraining? | **Done** | Yes (85% after ingestion); added intent-mix drift alert |
| E4 | Would a different embedding model retrieve better? | **Done** | all-mpnet-base-v2 is significantly better (+0.16 nDCG@10); switch recommended, deferred pending a re-baseline |
| E5 | Where should the RESOLVE threshold be? | **Done on AI ratings** (provisional) | Keep 0.75; 0.90 gives 100% precision at 42% coverage; the real fix is wrong-scenario detection |

---

## E1: Retrieval method comparison (MUST RUN)

**Script:** `python experiments/e1_retrieval.py` → `experiments/results/e1_retrieval.json`

**Hypothesis.** Hybrid (pgvector cosine + PostgreSQL FTS, fused with weighted RRF) beats pure semantic and pure
lexical retrieval, and all of them beat today's keyword search.

**Setup.** 200 held-out queries (`retrieval_eval.jsonl`) whose problem phrasing never appears in the corpus;
2,043 tickets + 40 KB articles; graded qrels (2 = same root-cause scenario, 1 = same intent + overlapping product).
Configurations: `keyword_baseline` (agent-style: 2 most specific words, ILIKE, newest first), `lexical_and`
(`plainto_tsquery`), `lexical_or` (OR-of-terms `tsquery`, `ts_rank`), `semantic` (ivfflat), and hybrid with
semantic:lexical weights 50:50, 70:30, 30:70. All with `kb_min_slots=0` so raw ranking is compared.
Metrics: nDCG@10, MRR@20, P@5, Hit@5, capped Recall@10, KB-Recall@10, latency p50/p95, bootstrap 95% CIs and
paired-bootstrap p-values for the best hybrid against each single method.

**Success criterion (plan).** Hybrid beats both single methods on ≥ 2 of 4 headline metrics; otherwise adjust
weights or use the best single method.

**Weight selection (dev split, 199 queries from `test.jsonl`, judgments built by the same rule as the eval qrels).**

| semantic : lexical | 30:70 | 50:50 | 60:40 | 70:30 | **80:20** | 90:10 |
|---|---:|---:|---:|---:|---:|---:|
| dev nDCG@10 | 0.444 | 0.494 | 0.549 | 0.557 | **0.563** | 0.559 |
| dev MRR | 0.640 | 0.661 | 0.662 | 0.655 | 0.651 | 0.618 |

60:40 through 90:10 form a plateau; 80:20 was selected and is now the production default (`RRF_SEMANTIC_WEIGHT`).

**Results (200 held-out queries, top-20, `kb_min_slots=0`).**

| Configuration | nDCG@10 | MRR | P@5 | Hit@5 | Recall@10 (capped) | KB-Recall@10 | latency p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| keyword_baseline (today's practice) | 0.223 | 0.270 | 0.210 | 0.405 | 0.199 | 0.000 | 49 / 88 ms |
| lexical_and (`plainto_tsquery`) | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 50 / 65 ms |
| lexical_or (OR-of-terms FTS) | 0.274 | 0.452 | 0.270 | 0.565 | 0.242 | 0.005 | 172 / 268 ms |
| semantic (pgvector) | 0.557 | 0.600 | 0.545 | 0.715 | **0.559** | 0.085 | 61 / 88 ms |
| hybrid 50:50 | 0.518 | **0.684** | 0.546 | **0.780** | 0.476 | **0.500** | 179 / 243 ms |
| **hybrid 80:20 (selected on dev)** | **0.572** | 0.657 | **0.561** | 0.760 | 0.558 | 0.190 | 189 / 646 ms |

nDCG@10 95% bootstrap CIs: semantic [0.503, 0.605], hybrid 80:20 [0.520, 0.618], keyword [0.183, 0.264].

Hybrid 80:20 vs semantic (paired bootstrap, one-sided):

| metric | Δ | p |
|---|---:|---:|
| MRR | +0.057 | 0.0005 |
| Hit@5 | +0.045 | 0.010 |
| nDCG@10 | +0.015 | 0.046 |
| P@5 | +0.016 | 0.071 (n.s.) |
| Recall@10 (capped) | −0.001 | 0.57 (n.s.) |

Against lexical_or and the keyword baseline, every metric improves by +0.20 to +0.39 (p < 0.001).

**Interpretation.**
* **Keyword search, the status quo, finds a highly relevant ticket in the top 5 for only 40.5% of complaints;
  hybrid does for 76%.** This is the problem statement's claim, measured: complaints phrased differently from past
  tickets are missed by keywords.
* **`plainto_tsquery` returns nothing for every query.** It ANDs all terms, and a 30-word complaint never matches a
  ticket on every term. This is why the lexical side uses an OR-of-terms query with `ts_rank` ordering.
* **Hybrid's gain over semantic-only is modest and concentrated at the top of the list** (MRR, Hit@5): exact
  tokens (error codes, "LOS", "PAC", "NAT") promote the right ticket to rank 1. Deeper recall is unchanged.
* The weight is a real trade-off: more lexical weight surfaces KB articles (KB-Recall@10 0.50 at 50:50 vs 0.19 at
  80:20) but dilutes ticket precision. Production resolves this with `RETRIEVAL_KB_MIN_SLOTS=2` instead of a lower weight.
* Cost: about +120 ms median latency, from the OR-term FTS query over many matching rows. The p95 of 646 ms looks like
  run-to-run noise on a shared laptop (the 50:50 run, which issues identical SQL, had a p95 of 243 ms).

**Decision.** Hybrid retrieval with RRF, semantic:lexical = 0.8:0.2, OR-term lexical query, plus 2 guaranteed KB slots.

**A bug this experiment caught (v1 → v2).** The first run (`e1_retrieval_v1_per_table_rrf.json`) fused
rankings *per table* and then interleaved tickets and KB articles by RRF score. Because RRF depends only on rank,
the #1 of 40 KB articles tied the #1 of 2,043 tickets, so roughly half of every top-10 was KB articles, most of
them for other scenarios. v1 hybrid had nDCG@10 0.30 and P@5 0.28, far below semantic, while showing KB-Recall@10
0.90 and MRR 0.72. The fix builds one semantic and one lexical ranking across both tables and fuses once. The v1
file is kept for transparency.

---

## E2: Cross-encoder reranking (MUST RUN)

**Script:** `python experiments/e2_reranking.py` → `experiments/results/e2_reranking.json`

**Hypothesis.** Reranking the hybrid top-30 with `cross-encoder/ms-marco-MiniLM-L-6-v2` improves P@5 by more
than 0.05 (absolute) with less than 2 s added latency.

**Decision rule.** Enable `USE_RERANKING` only if both conditions hold. Until E2 runs, reranking is **off**
(the default the plan prescribes).

**Caveat to check when reading results.** The MS MARCO cross-encoder is trained for web question → passage
relevance, not complaint → resolved-ticket similarity, so a gain is not a foregone conclusion.

**Setup.** Same 200 held-out queries; hybrid 0.8:0.2 retrieves 30 candidates, the cross-encoder rescores them
(document text truncated to 2,000 chars), top 10 kept. Baseline: hybrid top-10. `kb_min_slots=0`. CPU only
(4-core laptop), nothing else running.

**Results.**

| | nDCG@10 | MRR | P@5 | P@10 | Hit@1 | Hit@5 | Recall@10 (capped) | KB-Recall@10 | latency mean / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 0.572 | 0.654 | 0.561 | 0.558 | 0.565 | 0.760 | 0.558 | 0.190 | 258 / 536 ms |
| hybrid + rerank | **0.625** | **0.723** | **0.613** | **0.607** | **0.660** | **0.800** | **0.607** | 0.060 | 2,980 / 4,949 ms |

* P@5 gain **+0.052 absolute** (+9.3% relative), paired-bootstrap p = 0.002. Quality bar met.
* Added latency **+2,722 ms mean**. Latency bar (< 2,000 ms) failed.
* The reranker also pushes KB articles out of the top 10 (KB-Recall@10 0.19 → 0.06): it prefers short,
  complaint-like ticket texts over long procedural articles.

**Decision.** **Keep reranking disabled** (`USE_RERANKING=false`), per the pre-registered rule. The quality gain is
real, so this is a deployment question, not a dead end. On a GPU, or with fewer candidates or an ONNX/quantised
cross-encoder, the latency term could fall under budget; re-run E2 in that environment before enabling. If enabled,
keep `RETRIEVAL_KB_MIN_SLOTS` so KB articles are not lost.

---

## E3: Groundedness validation methods (MUST RUN) ✅

**Script:** `python experiments/e3_groundedness.py` → `experiments/results/e3_groundedness.json`
(first iteration kept as `e3_groundedness_v1.json`).

**Hypothesis.** Combining semantic similarity, NLI entailment/contradiction and lexical overlap detects
unsupported and contradicted claims better than any single signal.

**Data.** 200 (claim, source, label) items, 5 per wave-1 scenario:

| kind | label | n | how it is built |
|---|---|---:|---|
| paraphrase | supported | 80 | a **paraphrase** of a fix step present in the source (KB article or resolved ticket) |
| verbatim | supported | 40 | a fix step copied **verbatim** from a real corpus ticket (added in v2, see below) |
| unsupported | not supported | 40 | a real fix step from a *different* scenario of the *same* intent (hard negative, shared vocabulary) |
| contradicted | not supported | 40 | a claim that directly opposes the scenario's fix (e.g. "disable 5 GHz" when the fix enables it) |

**Protocol.** Split by scenario into dev (20 scenarios) and test (20 scenarios); no scenario in both. For each
configuration and method, thresholds are grid-searched on dev to maximise F1 for detecting **not-supported**
claims, then reported on test. NLI model: `cross-encoder/nli-deberta-v3-small`.

### Iteration: why there is a v1 and a v2

E3 v1 (160 items, no verbatim subset) found multi-method F1 0.804 and was adopted. The end-to-end generation
eval then exposed a failure E3 v1 couldn't see. Extractive drafts copy steps *verbatim* from sources, yet
validation called **88 of 100 drafts "contradicted"**. Root cause, from claim-level debugging:

* NLI models are trained on declarative statements, and fix steps are imperatives. Given a premise containing
  *"Enable QoS on the router to prioritise video calls…"* verbatim, DeBERTa returned entailment **0.02**.
  A *different* instruction ("Replace the SIM card") got contradiction **0.96**. For instructions, NLI conflates
  "a different action" with "the opposite action".
* v1 fed 400-char multi-sentence chunks as premises and took the **maximum** contradiction over every chunk of
  every cited source. A real draft cites 3–4 sources, so one confused pair was enough to flag a correct step.

v2 changes (`GroundednessChecker`): (1) **sentence-level premises**; (2) contradiction read from the premise
**most similar to the claim** (the one about the same action), not the maximum; (3) a **quotation rule**: a
claim that appears verbatim in a cited source is supported and skips NLI. The rule is sound because the whole
claim must be quoted, so "*Do not* restart the router" never matches "restart the router". A verbatim subset
was added to E3 so the failure is now part of the test.

### Results (test half; thresholds tuned on dev for each row)

| Config | Method | F1 (not-supported) | Precision | Recall | Accuracy | Contradiction recall | False alarms: paraphrase | False alarms: verbatim |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v1 chunks, max | similarity | 0.597 | 0.426 | 1.000 | 0.460 | 0.00 | 0.95 | 0.80 |
| v1 chunks, max | lexical | 0.739 | 0.654 | 0.850 | 0.760 | 0.00 | 0.45 | 0.00 |
| v1 chunks, max | NLI | 0.635 | 0.465 | 1.000 | 0.540 | 0.80 | 0.68 | **0.95** |
| v1 chunks, max | multi | 0.736 | 0.591 | 0.975 | 0.720 | 0.75 | 0.48 | **0.40** |
| v2 sentences, top-sim, quote | similarity | 0.694 | 0.781 | 0.625 | 0.780 | 0.00 | 0.18 | 0.00 |
| v2 sentences, top-sim, quote | lexical | 0.739 | 0.654 | 0.850 | 0.760 | 0.00 | 0.45 | 0.00 |
| v2 sentences, top-sim, quote | NLI | 0.897 | 0.830 | 0.975 | 0.910 | 0.70 | 0.20 | 0.00 |
| **v2 sentences, top-sim, quote** | **multi** | **0.918** | **0.867** | **0.975** | **0.930** | 0.70 | **0.15** | **0.00** |

With `weakly_supported` counted as supported, v2 multi reaches F1 0.949 / accuracy 0.960 (1 false positive in 80).

Mean signals by kind (all 200 items):

| kind | v1 entailment | **v2 entailment** | v1 contradiction | v2 contradiction | semantic (v2) | lexical |
|---|---:|---:|---:|---:|---:|---:|
| paraphrase (supported) | 0.275 | **0.777** | 0.109 | 0.051 | 0.672 | 0.485 |
| verbatim (supported) | 0.058 | **0.972** | 0.459 | 0.004 | 0.904 | 1.000 |
| unsupported | 0.004 | 0.068 | 0.116 | 0.217 | 0.329 | 0.143 |
| contradicted | 0.002 | 0.001 | 0.655 | 0.507 | 0.616 | 0.415 |

**Findings.**
* **Similarity alone cannot detect contradictions.** Contradicting claims are about as similar to their source as
  supported paraphrases are (0.62 vs 0.67), because "disable 5 GHz" and "enable 5 GHz" share almost every token.
  Contradiction recall for similarity-only and lexical-only is 0.00 in both iterations.
* **NLI needs single-sentence premises.** Moving from chunks to sentences nearly triples entailment for correct
  paraphrases and makes NLI the strongest single signal (F1 0.635 → 0.897).
* **Multi-method is still best** (0.918), confirming the plan's design.
* The cost of v2: contradiction recall 0.75 → 0.70 (14 of 20 contradictions still labelled `contradicted`, and 19
  of 20 detected as not-supported in some form).
* Scoring cost: 376 ms per claim with full NLI (460 ms in v1); quoted claims skip NLI in production.

**Decision.** Multi-method v2 in production. Thresholds from the dev half: `sim 0.60 / sim_weak 0.45 / entail 0.30
/ entail_weak 0.15 / lexical 0.50 / lexical_weak 0.30 / contradiction 0.50` (`app/config.py`). `weakly_supported`
earns half credit in the groundedness score.

**Limitations.** 100 test items (one item ≈ 1 accuracy point). Supported claims are template paraphrases or
verbatim copies, not free-form LLM text; LLM-draft groundedness is measured in EVALUATION.md §4. The v2 design was
motivated by a failure seen in the generation eval, and its thresholds were then chosen on E3's dev half only.

---

## E4′: Evolving ticket classes ✅

**Script:** `python experiments/evolving_classes.py --extractive` → `experiments/results/evolving_classes.json`.
Runs against the live database through the same services the API uses (`/ingestion`, `/taxonomy/intents`,
`/monitoring/drift` code paths). Extractive generation, so no LLM quota is used and the run is deterministic.

**Question.** A new issue type (`roaming_issue`: 3 scenarios, 221 tickets, 3 KB articles) starts arriving after
launch. (a) Does the system notice before anyone tells it? (b) After the class is registered and its tickets are
ingested through the normal ingestion path, with no retraining or redeploy, does the system handle it?

**Procedure.** 60 in-distribution complaints are sent and back-dated to form the drift baseline; 60 held-out roaming
complaints are sent (phase A) and the drift report is read; the intent is registered and the wave-2 tickets and KB
articles are ingested; the same 60 complaints are sent again (phase C). Afterwards wave-2 rows are removed so the
other evaluations stay reproducible.

**Results.**

| | Phase A: class unknown | Phase C: after ingestion |
|---|---:|---:|
| Classified `roaming_issue` | 0% | **85.0%** |
| Flagged `unknown_intent` | 3.3% | 0% |
| Roaming KB article among sources | 0% | **86.7%** |
| Decisions RESOLVE / REVIEW / ESCALATE | 29 / 30 / 1 | 39 / 21 / 0 |
| Mean nearest-neighbour similarity | 0.632 | 0.710 |

Ingestion: 221 tickets + 3 KB articles inserted, 445 embeddings, 0 errors, `new_categories = ["roaming_issue"]`.

**Drift detection (phase A vs baseline).**

| Signal | Baseline | Recent | Alert? |
|---|---:|---:|---|
| Unknown-intent rate | 0.017 | 0.033 | no (threshold 0.15) |
| Mean nearest-neighbour similarity | 0.644 | 0.632 | no (threshold 0.45) |
| **Intent-mix TVD** | | **0.517** | **yes** (threshold 2.5/√60 = 0.323) |

The first run of this experiment raised **no alert**: the monitor computed TVD but only alerted on unknown rate
and similarity, which barely move because roaming complaints resemble mobile tickets. The new class didn't look
*unfamiliar*; it showed up as an unusual **mix** of familiar classes (connectivity 15% → 42%, outage 13% → 25%,
billing 17% → 28%). A size-aware TVD alert was added. Its threshold approximates the 99th percentile of TVD between two
normal windows (measured p99: 0.467 / 0.317 / 0.217 / 0.138 for windows of 30 / 60 / 120 / 240), and the re-run
fires: *"intent mix shifted: TVD 0.517 > 0.323 … a new issue type may be landing in existing classes"*.

**Findings.**
* **Adaptation works without retraining:** register the intent, ingest tickets, and 85% of the new class is recognised
  with its KB article retrieved 87% of the time.
* **Before ingestion the system is overconfident:** 48% of novel complaints are RESOLVEd with drafts grounded in the
  wrong tickets (extractive generator; EVALUATION.md §4.1, "grounded but wrong"). An LLM generator that may decline
  when sources don't fit lowers this to 22% (EVALUATION.md §4.2). Per-request signals can't see this; the
  traffic-level intent-mix alert can, after about 60 requests. That is the operational backstop: the alert triggers a
  review, the reviewer registers the class, and ingestion fixes it.
* Cost of the new class: in-distribution intent accuracy drops 0.724 → 0.682 (offline eval), because roaming tickets
  attract some mobile complaints of other classes.

**Limitations.** One synthetic new class; the TVD coefficient was calibrated on this dataset's balanced class mix.
Real traffic has seasonal mix shifts, so the coefficient should be recalibrated from historical windows and
the alert treated as a review trigger, not an automatic action.

---

## Optional experiments (plan §35): status

| Plan ID | Topic | Status |
|---|---|---|
| E4 | Embedding model comparison | done (section below) |
| E5 | Confidence threshold tuning | done on AI ratings, provisional (section below) |
| E6 | LLM temperature | not run: 4 temperatures × 25 cases = 100 extra LLM drafts (~350k tokens), about two days of the free Groq quota for the deployed model, which went to completing the main LLM evaluation |
| E7 | Context length (top-k sources) | not run: same quota constraint (4 settings × 25 cases); E1's deployed-retriever numbers show the scenario's KB article is in the top 8 for most queries |
| E8 | Caching | not run (Tier 3) |

## E5: Confidence threshold tuning (plan §35.2), on AI ratings

**Question.** What RESOLVE threshold balances precision (auto-resolved drafts that are safe) against coverage
(share of drafts auto-resolved)?

**Data.** (confidence, "safe to use as-is") pairs for the 50 drafts of the rating sheet, rated by an AI rater blind
to system outputs (EVALUATION.md §6.4; `experiments/results/ai_rater_eval.json`, `resolve_threshold_sweep`).
Coverage counts drafts at or above the threshold; policy rules (critical severity, unknown intent, injection) can
still send some of them to REVIEW, so coverage is an upper bound on automation.

| RESOLVE threshold | Coverage | Precision (safe) |
|---:|---:|---:|
| 0.55 - 0.65 | 96% | 75% |
| 0.70 | 94% | 74% |
| **0.75 (deployed)** | 92% | 76% |
| 0.80 | 88% | 77% |
| 0.85 | 68% | 82% |
| 0.90 | 42% | 100% |

Confidence AUROC for "safe" is 0.81, so the score does rank drafts, but precision only rises steeply above 0.85.

**Decision.** Keep 0.75 as the deployed default for now. If precision matters more than automation, 0.90 is the
setting this data supports (no unsafe draft auto-resolved, but fewer than half auto-resolved) and is a one-line
config change (`DECISION_RESOLVE_THRESHOLD`). Not adopted yet because (1) the ratings are AI-generated and n = 50,
so the precision estimates are wide; (2) the threshold treats the symptom: the unsafe auto-resolves are wrong-scenario
drafts with confidence 0.81-0.89, well grounded in the wrong sources, which a threshold cannot separate from good
drafts without discarding most of them. The structural fix is per-request wrong-scenario detection (EVALUATION.md
§9, item 1). Re-run this sweep on human ratings before changing the production threshold.

## E4: Embedding model comparison (plan §35.1)

**Hypothesis.** A larger or newer embedding model retrieves better than all-MiniLM-L6-v2; the plan's rule is to keep
MiniLM unless the improvement is significant.

**Procedure.** `experiments/e4_embeddings.py` → `experiments/results/e4_embeddings.json`. Same corpus as production
(2,043 tickets as "complaint resolution", 40 KB articles as "title\ncontent"), same 200 held-out queries and graded
judgments as E1. Each model embeds corpus and queries; ranking is exact cosine top-20 in memory, which isolates the
embedding model (no ivfflat approximation, no lexical fusion, no KB slots). Paired bootstrap against MiniLM.
Latencies are on the CPU-limited container (2 cores), so compare them relatively.

| Model | Dims | nDCG@10 | MRR | P@5 | Hit@5 | Recall@10 (capped) | Query encode p50 | Corpus encode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2 (deployed) | 384 | 0.554 | 0.600 | 0.543 | 0.715 | 0.555 | 123 ms | 117 s |
| **all-mpnet-base-v2** | 768 | **0.716** | **0.752** | **0.704** | **0.870** | **0.721** | 237 ms | 775 s |
| BAAI/bge-small-en-v1.5 | 384 | 0.501 | 0.551 | 0.485 | 0.755 | 0.518 | 148 ms | 199 s |

* mpnet vs MiniLM: +0.161 nDCG@10, +0.152 MRR, +0.161 P@5, +0.155 Hit@5, +0.165 Recall@10, all p < 0.001 (one-sided
  paired bootstrap, 2,000 resamples). It is the only configuration in this project whose capped Recall@10 clears the
  plan's 0.70 target, and the gain is larger than anything hybrid fusion or reranking achieved (E1, E2).
* bge-small vs MiniLM: significantly worse on nDCG@10 (-0.053, p = 0.018) and P@5 (-0.058, p = 0.016), slightly
  better Hit@5 (+0.04, n.s.). Run without bge's optional query instruction, which may cost it some quality.
* Exact in-memory MiniLM (nDCG@10 0.554) matches pgvector ivfflat in E1 (0.557): the ANN index costs nothing here.
* Cost of mpnet: 2x dimensions (index and storage), about 2x query-encode and 6-7x corpus-encode time on CPU, and a
  larger model in memory.

**Decision.** By the plan's rule mpnet should replace MiniLM, and it is the recommended next change. It was **not
switched in this submission**, because the switch is a re-baseline, not a config edit: the vector columns change to
768 dimensions, the corpus is re-embedded and the ivfflat indexes rebuilt, and several thresholds were tuned on
MiniLM's cosine scale (groundedness similarity in E3, evidence-sufficiency relevance floors, the confidence
normaliser), so E3's threshold tuning and every downstream evaluation (understanding k-NN, E1, E2, the 100-case LLM
run, about one day of free-tier LLM quota) must be re-run to keep the reported numbers consistent with what is
deployed. Better semantic neighbours may also lift intent F1, which uses the same embeddings for k-NN.

