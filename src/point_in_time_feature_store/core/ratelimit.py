"""In-process token-bucket rate limiting.

Each client key gets a bucket holding at most ``burst`` tokens that refills at ``rate`` tokens per
second, so over any interval of length T a key is admitted at most ``burst + rate * T`` times.
State is per process and bounded by ``max_keys`` (least recently used keys are evicted); a
multi-replica deployment needs a shared store instead, which is a documented non-goal here.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["RateDecision", "RateLimiter", "TokenBucket"]


class TokenBucket:
    """A single token bucket driven by an external clock value."""

    __slots__ = ("capacity", "rate", "tokens", "updated")

    def __init__(self, rate: float, capacity: float, now: float) -> None:
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.updated = now

    def try_acquire(self, now: float, cost: float = 1.0) -> tuple[bool, float]:
        """Take ``cost`` tokens if available. Returns (allowed, seconds until it would be)."""
        elapsed = max(0.0, now - self.updated)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.updated = max(self.updated, now)
        if self.tokens >= cost:
            self.tokens -= cost
            return True, 0.0
        return False, (cost - self.tokens) / self.rate


@dataclass(frozen=True, slots=True)
class RateDecision:
    allowed: bool
    retry_after_seconds: int
    remaining: int


class RateLimiter:
    """Thread-safe map of client key -> token bucket with LRU eviction."""

    def __init__(
        self,
        rate: float,
        burst: int,
        *,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_keys < 1:
            raise ValueError("max_keys must be >= 1")
        self._rate = rate
        self._burst = float(burst)
        self._max_keys = max_keys
        self._clock = clock
        self._buckets: OrderedDict[str, TokenBucket] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._buckets)

    def check(self, key: str) -> RateDecision:
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(self._rate, self._burst, now)
                self._buckets[key] = bucket
                if len(self._buckets) > self._max_keys:
                    self._buckets.popitem(last=False)
            else:
                self._buckets.move_to_end(key)
            allowed, wait = bucket.try_acquire(now)
            return RateDecision(
                allowed=allowed,
                retry_after_seconds=0 if allowed else max(1, math.ceil(wait)),
                remaining=int(bucket.tokens),
            )
