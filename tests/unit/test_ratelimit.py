from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from point_in_time_feature_store.core.ratelimit import RateLimiter, TokenBucket

pytestmark = pytest.mark.unit


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_burst_is_admitted_then_refills_at_rate() -> None:
    clock = FakeClock()
    limiter = RateLimiter(rate=1.0, burst=3, clock=clock)
    assert [limiter.check("a").allowed for _ in range(3)] == [True, True, True]
    denied = limiter.check("a")
    assert not denied.allowed
    assert denied.retry_after_seconds == 1
    clock.now = 1.0
    assert limiter.check("a").allowed


def test_keys_have_independent_buckets() -> None:
    limiter = RateLimiter(rate=0.001, burst=1, clock=FakeClock())
    assert limiter.check("a").allowed
    assert not limiter.check("a").allowed
    assert limiter.check("b").allowed


def test_tracked_keys_are_bounded() -> None:
    limiter = RateLimiter(rate=1.0, burst=1, max_keys=2, clock=FakeClock())
    for key in "abcde":
        limiter.check(key)
    assert len(limiter) == 2


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        TokenBucket(rate=0.0, capacity=1.0, now=0.0)
    with pytest.raises(ValueError, match="max_keys"):
        RateLimiter(rate=1.0, burst=1, max_keys=0)


def test_clock_going_backwards_does_not_mint_tokens() -> None:
    bucket = TokenBucket(rate=1.0, capacity=1.0, now=10.0)
    assert bucket.try_acquire(10.0)[0]
    assert not bucket.try_acquire(5.0)[0]


@given(
    rate=st.floats(min_value=0.1, max_value=50.0),
    burst=st.integers(min_value=1, max_value=20),
    gaps=st.lists(st.floats(min_value=0.0, max_value=2.0), min_size=1, max_size=150),
)
def test_admissions_never_exceed_the_token_bucket_bound(
    rate: float, burst: int, gaps: list[float]
) -> None:
    clock = FakeClock()
    limiter = RateLimiter(rate=rate, burst=burst, clock=clock)
    admitted: list[float] = []
    for gap in gaps:
        clock.now += gap
        if limiter.check("k").allowed:
            admitted.append(clock.now)
    for i, start in enumerate(admitted):
        for j in range(i, len(admitted)):
            window = admitted[j] - start
            bound = burst + rate * window
            assert j - i + 1 <= bound + 1e-6 * (1 + bound)
