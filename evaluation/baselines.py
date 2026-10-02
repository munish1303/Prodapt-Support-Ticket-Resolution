"""Baselines for comparison.

KeywordSearchBaseline emulates today's agent workflow (the problem statement):
agents type a couple of keywords ("router", "billing") into the ticket tool, which
returns tickets containing all of them, newest first, with no relevance ranking.
Keywords are the complaint's most specific content words (highest IDF over the
corpus), mimicking what an agent would pick out.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from sqlalchemy import text

from app.utils.text import STOPWORDS

_WORD = re.compile(r"[a-z][a-z0-9-]{2,}")


class KeywordSearchBaseline:
    def __init__(self, session_factory, n_keywords: int = 2):
        self.session_factory = session_factory
        self.n_keywords = n_keywords
        self.idf: dict[str, float] = {}

    async def fit_idf(self) -> None:
        async with self.session_factory() as session:
            rows = (
                await session.execute(text("SELECT complaint || ' ' || coalesce(resolution,'') FROM tickets"))
            ).all()
        df: Counter[str] = Counter()
        for (doc,) in rows:
            df.update(set(_WORD.findall(doc.lower())))
        n = len(rows) or 1
        self.idf = {w: math.log(n / (1 + c)) for w, c in df.items()}

    def keywords(self, complaint: str) -> list[str]:
        words = {w for w in _WORD.findall(complaint.lower()) if w not in STOPWORDS and w in self.idf}
        return sorted(words, key=lambda w: -self.idf[w])[: self.n_keywords]

    async def search(self, complaint: str, top_k: int = 20) -> list[str]:
        kws = self.keywords(complaint)
        if not kws:
            return []
        async with self.session_factory() as session:
            for op in (" AND ", " OR "):  # all keywords first; fall back to any keyword
                clauses: list[str] = []
                params: dict[str, object] = {"k": top_k}
                for i, kw in enumerate(kws):
                    clauses.append(f"(complaint ILIKE :kw{i} OR resolution ILIKE :kw{i})")
                    params[f"kw{i}"] = f"%{kw}%"
                rows = (
                    await session.execute(
                        text(
                            f"SELECT ticket_id FROM tickets WHERE {op.join(clauses)} ORDER BY created_at DESC LIMIT :k"
                        ),
                        params,
                    )
                ).all()
                if rows:
                    return [r[0] for r in rows]
        return []
