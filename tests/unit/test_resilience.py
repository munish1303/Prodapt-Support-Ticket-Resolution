"""Circuit breaker state machine (closed -> open -> half-open -> closed/open)."""

from __future__ import annotations

import asyncio

import pytest

from app.core.resilience import CLOSED, HALF_OPEN, OPEN, CircuitBreaker, CircuitOpenError


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Dependency:
    def __init__(self) -> None:
        self.calls = 0
        self.healthy = False

    async def __call__(self) -> str:
        self.calls += 1
        if not self.healthy:
            raise OSError("connection refused")
        return "ok"


def make(threshold: int = 3, recovery: float = 30.0, **kw) -> tuple[CircuitBreaker, Clock, Dependency]:
    clock = Clock()
    breaker = CircuitBreaker("db", threshold, recovery, failure_types=(OSError,), clock=clock, **kw)
    return breaker, clock, Dependency()


async def test_opens_after_consecutive_failures_and_then_fails_fast():
    breaker, _, dep = make(threshold=3)
    for _ in range(3):
        with pytest.raises(OSError):
            await breaker.call(dep)
    assert breaker.state == OPEN and dep.calls == 3
    with pytest.raises(CircuitOpenError):
        await breaker.call(dep)
    assert dep.calls == 3  # the dependency was not touched while open


async def test_success_resets_the_failure_count():
    breaker, _, dep = make(threshold=3)
    for _ in range(2):
        with pytest.raises(OSError):
            await breaker.call(dep)
    dep.healthy = True
    assert await breaker.call(dep) == "ok"
    dep.healthy = False
    for _ in range(2):
        with pytest.raises(OSError):
            await breaker.call(dep)
    assert breaker.state == CLOSED


async def test_half_open_trial_success_closes_failure_reopens():
    breaker, clock, dep = make(threshold=2, recovery=30)
    for _ in range(2):
        with pytest.raises(OSError):
            await breaker.call(dep)
    clock.now = 31
    assert breaker.state == HALF_OPEN
    with pytest.raises(OSError):  # trial fails -> open again at once, even though threshold is 2
        await breaker.call(dep)
    assert breaker.state == OPEN
    clock.now = 62
    dep.healthy = True
    assert await breaker.call(dep) == "ok"
    assert breaker.state == CLOSED


async def test_unlisted_exceptions_do_not_trip_the_breaker():
    breaker, _, _ = make(threshold=1)

    async def bug() -> None:
        raise ValueError("our own bug")

    with pytest.raises(ValueError):
        await breaker.call(bug)
    assert breaker.state == CLOSED


async def test_half_open_allows_the_configured_number_of_concurrent_trials():
    breaker, clock, _ = make(threshold=1, recovery=10, half_open_max_calls=2)

    async def down() -> None:
        raise OSError

    with pytest.raises(OSError):
        await breaker.call(down)
    clock.now = 11
    gate = asyncio.Event()

    async def slow_ok() -> str:
        await gate.wait()
        return "ok"

    t1 = asyncio.create_task(breaker.call(slow_ok))
    t2 = asyncio.create_task(breaker.call(slow_ok))
    await asyncio.sleep(0)
    with pytest.raises(CircuitOpenError):  # a third concurrent call is rejected
        await breaker.call(slow_ok)
    gate.set()
    assert await t1 == "ok" and await t2 == "ok"
    assert breaker.state == CLOSED
