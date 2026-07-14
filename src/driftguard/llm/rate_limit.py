from __future__ import annotations

import threading
import time
from typing import Callable


class RequestRateLimiter:
    """Thread-safe, process-local limiter shared by provider instances."""

    def __init__(
        self,
        requests_per_second: float | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if requests_per_second is not None and requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.interval = 0.0 if requests_per_second is None else 1.0 / requests_per_second
        self._clock, self._sleeper = clock, sleeper
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.interval == 0:
            return
        with self._lock:
            now = self._clock()
            delay = max(0.0, self._next_allowed - now)
            if delay:
                self._sleeper(delay)
                now = self._clock()
            self._next_allowed = max(now, self._next_allowed) + self.interval
