"""Tests for the backend-neutral callable distribution wrapper."""

import math
import random

import pytest

from baldr import CallableDistribution, distribution


def test_callable_distribution_preserves_functions_and_metadata() -> None:
    wrapped = CallableDistribution(
        ppf=lambda q: 2.0 * q,
        pdf=lambda x: 0.5 if 0.0 <= x <= 2.0 else 0.0,
        logpdf=lambda x: -math.log(2.0) if 0.0 <= x <= 2.0 else -math.inf,
        cdf=lambda x: min(max(x / 2.0, 0.0), 1.0),
        mean=1.0,
    )

    assert wrapped.ppf(0.25) == 0.5
    assert wrapped.cdf(0.5) == 0.25
    assert wrapped.mean == 1.0
    assert wrapped.median == 1.0


def test_callable_distribution_does_not_estimate_mean() -> None:
    calls = 0

    def pdf(x):
        nonlocal calls
        calls += 1
        return x

    wrapped = CallableDistribution(
        ppf=lambda q: q,
        pdf=pdf,
        logpdf=lambda x: math.log(x),
        cdf=lambda x: x,
    )

    assert calls == 0
    assert wrapped.mean is None


def test_callable_distribution_random_variate_uses_supplied_rng() -> None:
    wrapped = CallableDistribution(
        ppf=lambda q: q,
        pdf=lambda x: x,
        logpdf=lambda x: math.log(x),
        cdf=lambda x: x,
        rng=random.Random(42),
    )

    assert wrapped.rv() == pytest.approx(0.6394267984578837)


def test_pbjam_lower_case_alias_remains_available() -> None:
    assert distribution is CallableDistribution


@pytest.mark.parametrize("name", ["ppf", "pdf", "logpdf", "cdf"])
def test_non_callable_functions_are_rejected(name) -> None:
    functions = {
        "ppf": lambda q: q,
        "pdf": lambda x: x,
        "logpdf": lambda x: x,
        "cdf": lambda x: x,
    }
    functions[name] = None

    with pytest.raises(TypeError, match=name):
        CallableDistribution(**functions)
