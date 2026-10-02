import json
from types import SimpleNamespace

import pytest

from app.core.llm import LLMUnavailable, OpenAICompatibleProvider


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
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

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
    import httpx
    import openai

    req = httpx.Request("POST", "http://localhost:1")
    err = openai.APIConnectionError(request=req)
    provider, completions = provider_with([err, err, err], monkeypatch)
    with pytest.raises(LLMUnavailable):
        await provider.generate_json([])
    assert completions.calls == 3


@pytest.mark.asyncio
async def test_non_retryable_status_raises_immediately(monkeypatch):
    import httpx
    import openai

    resp = httpx.Response(401, request=httpx.Request("POST", "http://localhost:1"))
    err = openai.AuthenticationError("bad key", response=resp, body=None)
    provider, completions = provider_with([err], monkeypatch)
    with pytest.raises(LLMUnavailable):
        await provider.generate_json([])
    assert completions.calls == 1
