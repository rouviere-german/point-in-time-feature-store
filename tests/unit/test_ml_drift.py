from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from point_in_time_feature_store.ml.drift import DriftMonitor, kolmogorov_survival, ks_2samp, psi

pytestmark = pytest.mark.unit

_samples = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=1, max_size=30
)


def _brute_force_ks(a: list[float], b: list[float]) -> float:
    points = sorted(set(a) | set(b))
    return max(
        abs(sum(x <= p for x in a) / len(a) - sum(x <= p for x in b) / len(b)) for p in points
    )


def test_identical_samples_show_no_drift() -> None:
    sample = np.random.default_rng(0).normal(size=1_000)
    assert psi(sample, sample) == pytest.approx(0.0, abs=1e-12)
    result = ks_2samp(sample, sample)
    assert (result.statistic, result.pvalue) == (0.0, 1.0)


def test_a_one_sigma_shift_is_detected() -> None:
    rng = np.random.default_rng(1)
    reference, shifted = rng.normal(size=2_000), rng.normal(loc=1.0, size=2_000)
    assert psi(reference, shifted) > 0.2
    assert ks_2samp(reference, shifted).pvalue < 1e-6


def test_kolmogorov_distribution_matches_classical_critical_values() -> None:
    assert kolmogorov_survival(1.358) == pytest.approx(0.05, abs=1e-3)
    assert kolmogorov_survival(1.628) == pytest.approx(0.01, abs=5e-4)
    assert kolmogorov_survival(0.1) == 1.0
    assert kolmogorov_survival(5.0) < 1e-20


@given(_samples, _samples)
def test_ks_statistic_matches_a_brute_force_oracle(a: list[float], b: list[float]) -> None:
    assert ks_2samp(a, b).statistic == pytest.approx(_brute_force_ks(a, b), abs=1e-12)


@given(
    st.lists(st.integers(-1000, 1000), min_size=1, max_size=30),
    st.lists(st.integers(-1000, 1000), min_size=1, max_size=30),
)
def test_ks_is_invariant_under_increasing_transforms(a: list[int], b: list[int]) -> None:
    original = ks_2samp(a, b).statistic
    transformed = ks_2samp([2 * x + 1 for x in a], [2 * x + 1 for x in b]).statistic
    assert transformed == original


@given(_samples, _samples)
def test_psi_is_never_negative(a: list[float], b: list[float]) -> None:
    assert psi(a, b) >= 0.0


def test_constant_reference_is_handled() -> None:
    assert psi([3.0] * 50, [3.0] * 50) == pytest.approx(0.0)
    assert psi([3.0] * 50, [4.0] * 50) > 0.2


@pytest.mark.parametrize(
    ("expected", "actual", "match"),
    [([], [1.0], "empty"), ([1.0, float("nan")], [1.0], "NaN"), ([1.0], [float("inf")], "NaN")],
)
def test_invalid_samples_are_rejected(
    expected: list[float], actual: list[float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        psi(expected, actual)
    with pytest.raises(ValueError, match="bins"):
        psi([1.0], [1.0], bins=1)


def test_monitor_reports_per_feature_drift() -> None:
    rng = np.random.default_rng(2)
    monitor = DriftMonitor({"a": rng.normal(size=2_000), "b": rng.normal(size=2_000)})
    report = monitor.check({"a": rng.normal(size=2_000), "b": rng.normal(loc=1.0, size=2_000)})
    assert report.drifted
    assert report.drifted_features == ("b",)
    with pytest.raises(ValueError, match="missing features: b"):
        monitor.check({"a": rng.normal(size=10)})
    with pytest.raises(ValueError, match="at least one feature"):
        DriftMonitor({})


def test_tiny_but_significant_shifts_need_a_minimum_effect_size() -> None:
    rng = np.random.default_rng(3)
    monitor = DriftMonitor({"x": rng.normal(size=20_000)})
    report = monitor.check({"x": rng.normal(loc=0.08, size=20_000)})
    feature = report.features[0]
    assert feature.ks_pvalue < 0.01
    assert feature.ks_statistic < monitor.ks_min_statistic
    assert not report.drifted
