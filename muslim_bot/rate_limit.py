from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque

from fastapi import HTTPException


class RateLimiter:
    """Sliding-window rate limiter, keyed by caller identity (e.g. client IP).

    In-memory by design: limits are per process. Behind a reverse proxy the
    caller key should come from X-Forwarded-For (see api.check_rate_limit).
    Set max_per_minute <= 0 to disable.
    """

    def __init__(self, max_per_minute: int, *, max_keys: int = 10_000) -> None:
        self.max_per_minute = max_per_minute
        self.max_keys = max_keys
        self._lock = threading.Lock()
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    def check(self, key: str) -> None:
        if self.max_per_minute <= 0:
            return
        now = time.monotonic()
        window_start = now - 60.0
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                hits = self._hits[key] = deque()
            while hits and hits[0] <= window_start:
                hits.popleft()
            if len(hits) >= self.max_per_minute:
                retry_after = int(hits[0] - window_start) + 1
                raise HTTPException(
                    status_code=429,
                    detail="Terlalu banyak permintaan. Coba lagi sebentar.",
                    headers={"Retry-After": str(retry_after)},
                )
            hits.append(now)
            self._hits.move_to_end(key)
            while len(self._hits) > self.max_keys:
                self._hits.popitem(last=False)
