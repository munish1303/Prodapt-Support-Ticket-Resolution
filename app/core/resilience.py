"""Circuit breaker for dependencies that can go down (the database).

When the dependency keeps failing, waiting for every request to time out makes things worse: requests pile up
and the pool stays exhausted. After `failure_threshold` consecutive failures the breaker *opens* and calls fail
immediately with `CircuitOpenError`. After `recovery_timeout_s` it lets one trial call through (*half-open*): success
closes it, failure opens it again. `half_open_max_calls` allows that many concurrent trial calls (a request that
makes two independent database calls needs both to get through).

Only exceptions listed in `failure_types` count as dependency failures; anything else (e.g. a bug in our own code)
is re-raised without tripping the breaker.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


class CircuitOpenError(RuntimeError):
    """Raised instead of calling the dependency while the breaker is open."""


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        failure_types: tuple[type[BaseException], ...] = (Exception,),
        clock: Callable[[], float] = time.monotonic,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_timeout_s = recovery_timeout_s
        self.failure_types = failure_types
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self.half_open_max_calls = max(1, half_open_max_calls)
        self._trials_in_flight = 0

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return CLOSED
        if self._clock() - self._opened_at >= self.recovery_timeout_s:
            return HALF_OPEN
        return OPEN

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        state = self.state
        if state == OPEN or (state == HALF_OPEN and self._trials_in_flight >= self.half_open_max_calls):
            raise CircuitOpenError(f"{self.name} circuit is open")
        trial = state == HALF_OPEN
        if trial:
            self._trials_in_flight += 1
        try:
            result = await fn()
        except self.failure_types:
            self._record_failure(trial)
            raise
        finally:
            if trial:
                self._trials_in_flight -= 1
        self._record_success()
        return result

    def _record_failure(self, trial: bool) -> None:
        self._failures += 1
        if trial or self._failures >= self.failure_threshold:
            if self._opened_at is None or trial:
                logger.warning("circuit opened", extra={"extra_fields": {"circuit": self.name}})
            self._opened_at = self._clock()

    def _record_success(self) -> None:
        if self._opened_at is not None:
            logger.info("circuit closed", extra={"extra_fields": {"circuit": self.name}})
        self._failures = 0
        self._opened_at = None
