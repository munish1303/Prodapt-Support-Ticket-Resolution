"""RAG generation (plan §21-22).

Two generators share one interface:
* LLMGenerator       - abstractive, citation-constrained JSON output from an LLM.
* ExtractiveGenerator - deterministic, no network: selects resolution actions that
  recur across the retrieved sources and cites every source containing them.
  Used as graceful-degradation fallback (LLM down / rate-limited / no key) and as
  a no-LLM baseline in evaluation.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict

from app.config import settings
from app.core.llm import LLMProvider, LLMUnavailable
from app.core.prompts import RESOLUTION_SYSTEM_PROMPT, build_resolution_user_prompt
from app.models.schemas import Document, GenerationResult, QueryMetadata
from app.utils.text import redact_pii, sanitize_for_prompt, split_sentences

logger = logging.getLogger(__name__)


class ContextBuilder:
    def build(self, documents: list[Document], max_chars: int | None = None) -> tuple[str, int]:
        """Return (context, n_sources_included). Sources keep their 1-based numbering."""
        max_chars = max_chars or settings.MAX_CONTEXT_CHARS
        parts: list[str] = []
        used = 0
        for idx, doc in enumerate(documents, start=1):
            if doc.doc_type == "kb_article":
                header = f"[Source {idx}] KB article {doc.id}: {doc.metadata.get('title', '')}"
            else:
                header = f"[Source {idx}] Resolved ticket {doc.id} (category={doc.metadata.get('category')})"
            block = f"{header}\n{sanitize_for_prompt(doc.text)}"
            if parts and used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n---\n".join(parts), len(parts)


class LLMGenerator:
    def __init__(self, llm: LLMProvider, context_builder: ContextBuilder | None = None):
        self.llm = llm
        self.context_builder = context_builder or ContextBuilder()
        self.name = llm.name

    async def generate(
        self, complaint: str, metadata: QueryMetadata, documents: list[Document], temperature: float | None = None
    ) -> GenerationResult:
        context, _ = self.context_builder.build(documents)
        messages = [
            {"role": "system", "content": RESOLUTION_SYSTEM_PROMPT},
            # Complaints go to an external API: redact PII first.
            {"role": "user", "content": build_resolution_user_prompt(redact_pii(complaint), metadata, context)},
        ]
        data = await self.llm.generate_json(messages, temperature=temperature)
        steps = data.get("resolution_steps") or []
        if not isinstance(steps, list):
            steps = [str(steps)]
        steps = [
            re.sub(r"^\s*(?:step\s*)?\d+[:.)]\s*", "", str(s), flags=re.I).strip() for s in steps if str(s).strip()
        ]
        return GenerationResult(
            resolution_steps=steps,
            summary=str(data.get("summary", "")),
            estimated_time=data.get("estimated_time"),
            generator=self.name,
            raw_text=str(data),
        )


_INTRO_RE = re.compile(r"^(customer reported|investigated|ticket raised|caller described|complaint:)", re.I)
_OUTCOME_RE = re.compile(
    r"(resolved|confirmed|ticket closed|closed without|follow-up|did not respond|first contact)", re.I
)
_PREFIX_RE = re.compile(r"^(advised customer to|walked customer through|agent action)\s*:\s*", re.I)
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+(.*)$")


def _action_sentences(doc: Document) -> list[str]:
    if doc.doc_type == "kb_article":
        return [m.group(1).strip() for line in doc.text.splitlines() if (m := _NUMBERED_RE.match(line))]
    resolution = doc.text.split("Resolution:", 1)[-1]
    actions = []
    for sent in split_sentences(resolution):
        if sent.lower().startswith("root cause") or _INTRO_RE.match(sent) or _OUTCOME_RE.search(sent):
            continue
        actions.append(_PREFIX_RE.sub("", sent).strip())
    return [a for a in actions if len(a) > 15]


def _key(sentence: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", sentence.lower()).strip()


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, max_steps: int = 5, min_support: int = 1):
        self.max_steps = max_steps
        self.min_support = min_support

    async def generate(
        self, complaint: str, metadata: QueryMetadata, documents: list[Document], temperature: float | None = None
    ) -> GenerationResult:
        support: dict[str, set[int]] = defaultdict(set)
        score: dict[str, float] = defaultdict(float)
        position: dict[str, list[float]] = defaultdict(list)
        text_for: dict[str, str] = {}
        for idx, doc in enumerate(documents, start=1):
            actions = _action_sentences(doc)
            weight = (1.5 if doc.doc_type == "kb_article" else 1.0) / idx**0.5
            for pos, action in enumerate(actions):
                k = _key(action)
                if not k:
                    continue
                text_for.setdefault(k, action.rstrip("."))
                if idx not in support[k]:
                    support[k].add(idx)
                    score[k] += weight
                    position[k].append(pos / max(len(actions), 1))
        candidates = [k for k in score if len(support[k]) >= self.min_support]
        top = sorted(candidates, key=lambda k: -score[k])[: self.max_steps]
        top.sort(key=lambda k: sum(position[k]) / len(position[k]))  # keep procedural order
        steps = [f"{text_for[k]} " + "".join(f"[{i}]" for i in sorted(support[k])[:4]) for k in top]
        n_tickets = sum(1 for d in documents if d.doc_type == "ticket")
        n_kb = len(documents) - n_tickets
        summary = (
            f"Steps that recur across {n_tickets} similar historical tickets and {n_kb} KB article(s)."
            if steps
            else "No actionable resolution steps found in the retrieved sources."
        )
        return GenerationResult(resolution_steps=steps, summary=summary, estimated_time=None, generator=self.name)


class GenerationService:
    def __init__(
        self,
        llm_generator: LLMGenerator | None,
        extractive: ExtractiveGenerator | None = None,
        fallback: bool | None = None,
    ):
        self.llm_generator = llm_generator
        self.extractive = extractive or ExtractiveGenerator()
        self.fallback = settings.LLM_FALLBACK_TO_EXTRACTIVE if fallback is None else fallback

    async def generate(self, complaint: str, metadata: QueryMetadata, documents: list[Document]) -> GenerationResult:
        if not documents:
            return GenerationResult([], "No relevant sources were retrieved.", None, generator="none")
        if self.llm_generator is not None and self.llm_generator.llm.available:
            try:
                try:
                    return await asyncio.wait_for(
                        self.llm_generator.generate(complaint, metadata, documents),
                        timeout=settings.LLM_REQUEST_DEADLINE_S,
                    )
                except TimeoutError as exc:
                    raise LLMUnavailable(f"LLM draft exceeded {settings.LLM_REQUEST_DEADLINE_S:.0f}s") from exc
            except LLMUnavailable as exc:
                logger.warning("LLM unavailable, falling back: %s", exc)
                if not self.fallback:
                    return GenerationResult(
                        [],
                        "Generation failed: LLM unavailable.",
                        None,
                        generator=self.llm_generator.name,
                        error=str(exc),
                    )
                result = await self.extractive.generate(complaint, metadata, documents)
                result.error = f"llm_unavailable: {exc}"
                return result
        return await self.extractive.generate(complaint, metadata, documents)
