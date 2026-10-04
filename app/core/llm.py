"""LLM provider abstraction.

`OpenAICompatibleProvider` talks to any endpoint implementing the OpenAI chat
completions API (Groq, Google Gemini's OpenAI endpoint, OpenAI, vLLM, Ollama),
selected purely through LLM_BASE_URL / LLM_MODEL / LLM_API_KEY.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """Raised when the LLM cannot be used (no key, network error, exhausted retries)."""


class LLMProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    async def generate_json(
        self, messages: list[dict[str, str]], temperature: float | None = None, max_tokens: int | None = None
    ) -> dict[str, Any]:
        """Return the model's JSON object response."""

    @property
    def available(self) -> bool:
        return True


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        max_retries: int | None = None,
        reasoning_effort: str | None = None,
    ):
        self.api_key = api_key if api_key is not None else settings.LLM_API_KEY
        # Only for reasoning models (e.g. gpt-oss): "low" keeps hidden reasoning from eating the token budget.
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else settings.LLM_REASONING_EFFORT
        self.base_url = base_url or settings.LLM_BASE_URL
        self.model = model or settings.LLM_MODEL
        self.max_retries = max_retries if max_retries is not None else settings.LLM_MAX_RETRIES
        self.name = f"llm:{self.model}"
        self._client = None
        if self.api_key:
            from openai import AsyncOpenAI

            try:  # openai>=3 is built on httpx2; older SDKs on httpx (same API)
                import httpx2 as httpx
            except ImportError:  # pragma: no cover
                import httpx  # type: ignore[no-redef]

            # Connection-level retries (connect errors / resets only; never re-sends a request that reached
            # the server). Optionally bind to 0.0.0.0 to force IPv4 on networks that advertise but drop IPv6.
            transport_kw: dict[str, Any] = {"retries": 3}
            if settings.LLM_FORCE_IPV4:
                transport_kw["local_address"] = "0.0.0.0"
            http_client = httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(**transport_kw),
                timeout=timeout_s or settings.LLM_TIMEOUT_S,
            )
            # Retries are handled here (with backoff that respects 429s), not by the SDK.
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout_s or settings.LLM_TIMEOUT_S,
                max_retries=0,
                http_client=http_client,
            )

    @property
    def available(self) -> bool:
        return self._client is not None

    async def generate_json(
        self, messages: list[dict[str, str]], temperature: float | None = None, max_tokens: int | None = None
    ) -> dict[str, Any]:
        if self._client is None:
            raise LLMUnavailable("LLM_API_KEY is not configured")
        import openai

        last_error: Exception | None = None
        waited = 0.0  # seconds spent waiting on rate limits in this call
        for attempt in range(self.max_retries):
            try:
                started = time.perf_counter()
                request: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature,
                    "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
                    "response_format": {"type": "json_object"},
                }
                if self.reasoning_effort:
                    request["extra_body"] = {"reasoning_effort": self.reasoning_effort}
                response = await self._client.chat.completions.create(**request)
                content = response.choices[0].message.content or "{}"
                usage = getattr(response, "usage", None)
                logger.info(
                    "llm_call",
                    extra={
                        "extra_fields": {
                            "model": self.model,
                            "latency_ms": int((time.perf_counter() - started) * 1000),
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(usage, "completion_tokens", None),
                        }
                    },
                )
                return json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = exc
            except openai.RateLimitError as exc:
                last_error = exc
                retry_after = _retry_after_seconds(exc)
                wait = retry_after if retry_after is not None else min(2 ** (attempt + 1), 30)
                # Someone is waiting on this request: a daily quota cannot recover within it, and a long per-minute
                # wait is worse than the extractive fallback. Give up at once in those cases.
                if _is_daily_quota(exc):
                    raise LLMUnavailable(f"LLM daily quota exhausted: {exc}") from exc
                if waited + wait > settings.LLM_MAX_RATE_LIMIT_WAIT_S:
                    raise LLMUnavailable(f"LLM rate-limited (retry after {wait:.0f}s): {exc}") from exc
                waited += wait
                await asyncio.sleep(wait)
                continue
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
                last_error = exc
            except openai.APIStatusError as exc:  # 4xx other than 429: not retryable
                raise LLMUnavailable(f"LLM request rejected: {exc.status_code} {exc.message}") from exc
            await asyncio.sleep(min(2**attempt, 30))
        raise LLMUnavailable(f"LLM failed after {self.max_retries} attempts: {last_error}")


def _is_daily_quota(exc: Exception) -> bool:
    """Groq/OpenAI-style 429 for a per-day limit (tokens or requests per day)."""
    text = str(exc).lower()
    return "per day" in text or "(tpd)" in text or "(rpd)" in text or "perday" in text


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    value = response.headers.get("retry-after")
    try:
        return min(float(value), 60.0) if value else None
    except ValueError:
        return None


def build_llm_provider() -> LLMProvider | None:
    if settings.LLM_PROVIDER == "extractive":
        return None
    return OpenAICompatibleProvider()
