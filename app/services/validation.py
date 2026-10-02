"""Citation and groundedness validation (plan §23-24).

Every generated step is a claim. For each claim we check:
1. Citation validity  - cited source numbers exist.
2. Groundedness       - is the claim supported by the sources it cites? Three signals:
     semantic   max cosine(claim, source chunk)                  (MiniLM embeddings)
     entailment max P(entailment | source chunk => claim)         (NLI cross-encoder)
     lexical    fraction of the claim's content words in source   (stemmed tokens)
   NLI also gives P(contradiction), which lets us flag claims that contradict a source.

Deviations from the plan's pseudo-code (see ARCHITECTURE.md):
* citation_coverage = claims with >=1 valid citation / all claims (the plan's
  formula always evaluated to 1.0).
* Uncited claims count as unsupported in the groundedness score (stricter).
* Similarity is computed against sentence chunks, not whole documents, because a
  short claim vs. a long document gives diluted similarity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.config import settings
from app.models.embeddings import EmbeddingService
from app.models.schemas import ClaimCheck, Document, ValidationResult
from app.utils.text import chunk_sentences, content_tokens, split_sentences

_CITATION_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
MIN_QUOTE_CHARS = 15


def normalise_for_quote(text: str) -> str:
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def extract_citations(step: str) -> tuple[str, list[int]]:
    """Return (claim text without markers, cited source numbers). Supports [1][2] and [1, 2]."""
    numbers: list[int] = []
    for group in _CITATION_GROUP.findall(step):
        numbers.extend(int(n) for n in re.split(r"\s*,\s*", group))
    claim = _CITATION_GROUP.sub("", step).strip()
    seen, ordered = set(), []
    for n in numbers:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return claim, ordered


class NLIBackend(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """Label probabilities ({'entailment', 'contradiction', 'neutral'}) for each (premise, hypothesis)."""
        ...


class NLIModel:
    """Wraps an NLI cross-encoder and returns label probabilities."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or settings.NLI_MODEL
        self.model = CrossEncoder(self.model_name, device="cpu")
        id2label = self.model.model.config.id2label
        self.labels = [id2label[i].lower() for i in range(len(id2label))]

    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        if not pairs:
            return []
        logits = np.asarray(self.model.predict(pairs, show_progress_bar=False, batch_size=16))
        if logits.ndim == 1:
            logits = logits[None, :]
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        return [dict(zip(self.labels, row.tolist())) for row in probs]


@dataclass
class GroundednessThresholds:
    sim: float = settings.GROUNDED_SIM_THRESHOLD
    sim_weak: float = settings.GROUNDED_SIM_WEAK_THRESHOLD
    entail: float = settings.GROUNDED_ENTAIL_THRESHOLD
    entail_weak: float = settings.GROUNDED_ENTAIL_WEAK_THRESHOLD
    lexical: float = settings.GROUNDED_LEXICAL_THRESHOLD
    lexical_weak: float = settings.GROUNDED_LEXICAL_WEAK_THRESHOLD
    contradiction: float = settings.GROUNDED_CONTRADICTION_THRESHOLD


def lexical_overlap(claim: str, source_text: str) -> float:
    claim_tokens = set(content_tokens(claim))
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & set(content_tokens(source_text))) / len(claim_tokens)


def decide_support(scores: dict[str, float], t: GroundednessThresholds, method: str = "multi") -> str:
    """Map signal scores to a support status.

    method="multi": the combined rule used in production.
    method="similarity" / "nli" / "lexical": single-signal baselines for Experiment 3.
    """
    sim, ent, con, lex = scores["semantic"], scores["entailment"], scores["contradiction"], scores["lexical"]
    if method == "multi" and scores.get("quoted"):
        return "supported"  # the whole claim appears verbatim in a cited source
    if method == "similarity":
        return "supported" if sim >= t.sim else "unsupported"
    if method == "lexical":
        return "supported" if lex >= t.lexical else "unsupported"
    if method == "nli":
        if con >= t.contradiction and ent < t.entail_weak:
            return "contradicted"
        return "supported" if ent >= t.entail else "unsupported"

    if con >= t.contradiction and ent < t.entail_weak:
        return "contradicted"
    if ent >= t.entail and sim >= t.sim_weak:
        return "supported"
    if sim >= t.sim and lex >= t.lexical and con < t.contradiction:
        return "supported"
    weak_signals = (sim >= t.sim_weak) + (ent >= t.entail_weak) + (lex >= t.lexical_weak)
    if weak_signals >= 2:
        return "weakly_supported"
    return "unsupported"


class GroundednessChecker:
    """Scores a claim against the sources it cites.

    premise_unit         "sentence" (default) or "chunk" (~400-char windows, the v1 behaviour).
    contradiction_agg    "top_similarity" (default): contradiction is read from the premise most similar
                         to the claim, i.e. the one about the same action; "max" (v1): maximum over all premises.
    use_quote            a claim quoted verbatim from a cited source is supported without NLI.
    Why: NLI models are trained on declarative statements. On imperative fix steps they often score a
    verbatim-supported step as neutral and a merely *different* action as a contradiction, so max-over-chunks
    produced false contradictions on real drafts (see EXPERIMENTS.md, E3 v2).
    """

    def __init__(
        self,
        embedder: EmbeddingService,
        nli: NLIBackend | None,
        thresholds: GroundednessThresholds | None = None,
        nli_top_premises: int | None = None,
        premise_unit: str | None = None,
        contradiction_agg: str | None = None,
        use_quote: bool | None = None,
    ):
        self.embedder = embedder
        self.nli = nli
        self.t = thresholds or GroundednessThresholds()
        self.nli_top_premises = nli_top_premises or settings.NLI_TOP_PREMISES
        self.premise_unit = premise_unit or settings.NLI_PREMISE_UNIT
        self.contradiction_agg = contradiction_agg or settings.NLI_CONTRADICTION_AGG
        self.use_quote = settings.GROUNDED_USE_QUOTE_MATCH if use_quote is None else use_quote

    def _units(self, text: str) -> list[str]:
        if self.premise_unit == "chunk":
            return chunk_sentences(text, 400)
        return split_sentences(text) or [text]

    def score(
        self, claim: str, sources: list[Document], _cache: dict | None = None, skip_nli_if_quoted: bool = True
    ) -> dict[str, float]:
        """Best score of each signal across the claim's cited sources (+ 'quoted' flag)."""
        cache = _cache if _cache is not None else {}
        claim_vec = self.embedder.encode(claim)
        norm_claim = normalise_for_quote(claim)
        best = {"semantic": 0.0, "entailment": 0.0, "contradiction": 0.0, "lexical": 0.0, "quoted": 0.0}
        premises: list[tuple[float, str]] = []
        for src in sources:
            key = (src.id, self.premise_unit)
            if key not in cache:
                units = self._units(src.text)
                cache[key] = (units, self.embedder.encode(units), normalise_for_quote(src.text))
            units, unit_vecs, norm_src = cache[key]
            if self.use_quote and len(norm_claim) >= MIN_QUOTE_CHARS and norm_claim in norm_src:
                best["quoted"] = 1.0
            sims = unit_vecs @ claim_vec
            best["semantic"] = max(best["semantic"], float(sims.max()))
            best["lexical"] = max(best["lexical"], lexical_overlap(claim, src.text))
            for i in np.argsort(-sims)[: self.nli_top_premises]:
                premises.append((float(sims[i]), units[i]))
        if self.nli is not None and premises and not (best["quoted"] and skip_nli_if_quoted):
            probs = self.nli.predict([(p, claim) for _, p in premises])
            best["entailment"] = max(pr.get("entailment", 0.0) for pr in probs)
            if self.contradiction_agg == "max":
                best["contradiction"] = max(pr.get("contradiction", 0.0) for pr in probs)
            else:
                top = max(range(len(premises)), key=lambda i: premises[i][0])
                best["contradiction"] = probs[top].get("contradiction", 0.0)
        return {k: round(v, 4) for k, v in best.items()}

    def check_claim(
        self, claim: str, sources: list[Document], method: str = "multi", _cache: dict | None = None
    ) -> tuple[str, dict]:
        scores = self.score(claim, sources, _cache)
        return decide_support(scores, self.t, method), scores


class ValidationService:
    def __init__(self, checker: GroundednessChecker):
        self.checker = checker

    def validate(self, steps: list[str], sources: list[Document]) -> ValidationResult:
        n_sources = len(sources)
        claims: list[ClaimCheck] = []
        issues: list[str] = []
        total_cites = invalid_cites = 0
        cache: dict = {}
        for step in steps:
            claim, cites = extract_citations(step)
            valid = [c for c in cites if 1 <= c <= n_sources]
            invalid = [c for c in cites if c not in valid]
            total_cites += len(cites)
            invalid_cites += len(invalid)
            if invalid:
                issues.append(f"Invalid citation(s) {invalid} in step: {claim[:60]}")
            if not valid:
                claims.append(ClaimCheck(claim, cites, invalid, "uncited"))
                issues.append(f"Uncited step: {claim[:60]}")
                continue
            status, scores = self.checker.check_claim(claim, [sources[c - 1] for c in valid], _cache=cache)
            if status in ("unsupported", "contradicted"):
                issues.append(f"Step {status} by cited sources: {claim[:60]}")
            claims.append(ClaimCheck(claim, cites, invalid, status, scores))

        n = len(claims)
        citation_accuracy = 1.0 - invalid_cites / total_cites if total_cites else 0.0
        citation_coverage = sum(1 for c in claims if c.support_status != "uncited") / n if n else 0.0
        supported = sum(1 for c in claims if c.support_status == "supported")
        weak = sum(1 for c in claims if c.support_status == "weakly_supported")
        groundedness = (supported + settings.WEAK_SUPPORT_CREDIT * weak) / n if n else 0.0
        return ValidationResult(
            claims, round(citation_accuracy, 4), round(citation_coverage, 4), round(groundedness, 4), issues
        )
