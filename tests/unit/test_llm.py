import json
from types import SimpleNamespace

import pytest

from app.core.llm import LLMUnavailable, OpenAICompatibleProvider


def _httpx_module():
    """openai>=3 builds on httpx2; its error classes expect httpx2 request/response objects."""
    try:
        import httpx2

        return httpx2
    except ImportError:  # pragma: no cover
        import httpx

        return httpx


@pytest.mark.asyncio
async def test_no_key_is_unavailable():
    provider = OpenAICompatibleProvider(api_key="")
    assert not provider.available
    with pytest.raises(LLMUnavailable):
        await provider.generate_json([{"role": "user", "content": "x"}])


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        msg = SimpleNamespace(content=outcome)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def provider_with(outcomes, monkeypatch):
    provider = OpenAICompatibleProvider(api_key="test", base_url="http://localhost:1", model="m", max_retries=3)
    completions = FakeCompletions(outcomes)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))  # type: ignore[assignment]

    async def no_sleep(_):
        return None

    monkeypatch.setattr("app.core.llm.asyncio.sleep", no_sleep)
    return provider, completions


@pytest.mark.asyncio
async def test_retries_on_bad_json_then_succeeds(monkeypatch):
    provider, completions = provider_with(["not json", json.dumps({"ok": True})], monkeypatch)
    assert await provider.generate_json([]) == {"ok": True}
    assert completions.calls == 2


@pytest.mark.asyncio
async def test_retries_on_connection_error_then_gives_up(monkeypatch):
    import openai

    httpx = _httpx_module()

    req = httpx.Request("POST", "http://localhost:1")
    err = openai.APIConnectionError(request=req)
    provider, completions = provider_with([err, err, err], monkeypatch)
    with pytest.raises(LLMUnavailable):
        await provider.generate_json([])
    assert completions.calls == 3


@pytest.mark.asyncio
async def test_non_retryable_status_raises_immediately(monkeypatch):
    import openai

    httpx = _httpx_module()

    resp = httpx.Response(401, request=httpx.Request("POST", "http://localhost:1"))
    err = openai.AuthenticationError("bad key", response=resp, body=None)
    provider, completions = provider_with([err], monkeypatch)
    with pytest.raises(LLMUnavailable):
        await provider.generate_json([])
    assert completions.calls == 1


def _rate_limit(message: str, retry_after: str | None = None):
    import openai

    httpx = _httpx_module()
    headers = {"retry-after": retry_after} if retry_after else {}
    resp = httpx.Response(429, headers=headers, request=httpx.Request("POST", "http://localhost:1"))
    return openai.RateLimitError(message, response=resp, body=None)


@pytest.mark.asyncio
async def test_daily_quota_fails_over_immediately(monkeypatch):
    err = _rate_limit("Rate limit reached for model on tokens per day (TPD): Limit 200000, Used 199252", "505")
    provider, completions = provider_with([err, json.dumps({"ok": True})], monkeypatch)
    with pytest.raises(LLMUnavailable, match="daily quota"):
        await provider.generate_json([])
    assert completions.calls == 1  # no waiting for a quota that cannot recover within the request


@pytest.mark.asyncio
async def test_long_rate_limit_wait_is_not_waited_out(monkeypatch):
    err = _rate_limit("Rate limit reached on tokens per minute (TPM)", "45")
    provider, completions = provider_with([err, json.dumps({"ok": True})], monkeypatch)
    with pytest.raises(LLMUnavailable, match="retry after 45s"):
        await provider.generate_json([])
    assert completions.calls == 1


@pytest.mark.asyncio
async def test_short_rate_limit_is_retried(monkeypatch):
    err = _rate_limit("Rate limit reached on tokens per minute (TPM)", "2")
    provider, completions = provider_with([err, json.dumps({"ok": True})], monkeypatch)
    assert await provider.generate_json([]) == {"ok": True}
    assert completions.calls == 2
