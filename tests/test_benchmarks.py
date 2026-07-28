"""Tests for the benchmark controls and runner."""

import math

import pytest

from baldr.benchmarks.implementations import (
    BenchmarkImplementation,
    scalar_normal_logpdf,
)
from baldr.benchmarks.runner import benchmark


def test_scalar_normal_logpdf_matches_definition() -> None:
    """The dependency-free control has the expected standard-normal value."""

    expected = -0.5 * (0.25**2 + math.log(2.0 * math.pi))
    assert scalar_normal_logpdf(0.25) == pytest.approx(expected)


def test_runner_reports_construction_first_and_warm_calls() -> None:
    """The runner keeps setup and steady-state costs separate."""

    implementation = BenchmarkImplementation(
        name="test",
        backend="scalar",
        factory=lambda: scalar_normal_logpdf,
        supports_scalar=True,
        supports_array=False,
    )
    report = benchmark((implementation,), sizes=(1,), repeat=2, number=3)

    assert not report["skipped"]
    result = report["results"][0]
    assert result["implementation"] == "test"
    assert result["size"] == 1
    assert result["construction_seconds"] >= 0.0
    assert result["first_call_seconds"] >= 0.0
    assert result["median_seconds"] >= 0.0
