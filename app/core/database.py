"""Async database engine and session management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def normalize_database_url(url: str) -> tuple[str, dict[str, Any]]:
    """Accept the URL forms hosted Postgres providers hand out (e.g. Neon on Vercel:
    ``postgresql://user:pw@host/db?sslmode=require&channel_binding=require``) and turn them into what the asyncpg
    driver understands. Returns (url, connect_args)."""
    parts = urlsplit(url)
    scheme = parts.scheme
    if scheme in ("postgres", "postgresql"):
        scheme = "postgresql+asyncpg"
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key == "sslmode":  # libpq name; asyncpg calls it ssl
            query.append(("ssl", value))
        elif key == "channel_binding":  # libpq-only option, not supported by asyncpg
            continue
        else:
            query.append((key, value))
    connect_args: dict[str, Any] = {}
    if "-pooler." in (parts.hostname or ""):
        # PgBouncer in transaction mode (Neon's pooled endpoint) cannot keep asyncpg's prepared statements.
        connect_args["statement_cache_size"] = 0
        query.append(("prepared_statement_cache_size", "0"))
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)), connect_args


def get_engine() -> AsyncEngine:
    global _engine, _session_factory
    if _engine is None:
        url, connect_args = normalize_database_url(settings.DATABASE_URL)
        _engine = create_async_engine(
            url,
            connect_args=connect_args,
            echo=False,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _session_factory is not None
    return _session_factory


async def get_db() -> AsyncIterator[AsyncSession]:
    async with get_session_factory()() as session:
        yield session


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine, _session_factory = None, None
