"""Text utilities: tokenisation, sentence splitting, PII redaction, injection heuristics."""

from __future__ import annotations

import hashlib
import re

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers him his how i if in into is it its itself just me more most my myself no nor
    not now of off on once only or other our ours out over own same she should so some such than that the
    their theirs them then there these they this those through to too under until up very was we were what
    when where which while who whom why will with would you your yours step customer please also ive im
    i've i'm it's dont don't can't cant""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def simple_stem(token: str) -> str:
    """Very small suffix stripper; enough to match 'restarting'/'restart', 'routers'/'router'."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def content_tokens(text: str) -> list[str]:
    return [simple_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s and len(s.strip()) > 3]


def chunk_sentences(text: str, max_chars: int = 400) -> list[str]:
    """Group consecutive sentences into chunks of at most ~max_chars."""
    chunks, current = [], ""
    for sent in split_sentences(text):
        if current and len(current) + len(sent) + 1 > max_chars:
            chunks.append(current)
            current = sent
        else:
            current = f"{current} {sent}".strip()
    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


# --- PII ----------------------------------------------------------------
_PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD]"),
    (re.compile(r"(?:\+?\d{1,3}[ -]?)?(?:\(?\d{2,5}\)?[ -]?)\d{3,4}[ -]?\d{3,4}\b"), "[PHONE]"),
    (re.compile(r"\b(?:account|acct|a/c)\s*(?:no\.?|number|#)?\s*[:#]?\s*\d{5,}\b", re.I), "[ACCOUNT]"),
]


def redact_pii(text: str) -> str:
    for pattern, token in _PII_PATTERNS:
        text = pattern.sub(token, text)
    return text


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- Prompt injection heuristics -----------------------------------------
_INJECTION_RE = re.compile(
    r"ignore (?:all |any )?(?:previous|prior|above) instructions|disregard (?:the )?(?:system|previous)|"
    r"you are now|system prompt|reveal your (?:prompt|instructions)|</?(?:system|assistant)>",
    re.I,
)


def looks_like_prompt_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


def sanitize_for_prompt(text: str) -> str:
    """Neutralise tag-like sequences so user text cannot close our prompt delimiters."""
    return text.replace("<", "‹").replace(">", "›")
