"""Tests for the dependency-free scalar backend."""

import math

import pytest

from baldr.scalar import (
    Beta,
    DiscreteUniform,
    Exponential,
    Normal,
    TruncatedNormal,
    TruncatedPowerLaw,
    TruncatedSine,
    Uniform,
    beta,
    normal,
    randint,
    truncsine,
    uniform,
)


@pytest.mark.parametrize(
    ("distribution", "probabilities"),
    [
        (Normal(loc=2.0, scale=3.0), (1e-10, 0.01, 0.5, 0.99, 1 - 1e-10)),
        (Uniform(loc=-2.0, scale=5.0), (0.0, 0.1, 0.5, 0.9, 1.0)),
        (Beta(a=0.4, b=3.2, loc=-1.0, scale=4.0), (1e-4, 0.01, 0.5, 0.99)),
        (Exponential(scale=2.5), (0.0, 0.1, 0.5, 0.99)),
        (
            TruncatedNormal(loc=0.5, scale=1.2, low=-1.0, high=2.0),
            (0.0, 0.01, 0.5, 0.99, 1.0),
        ),
        (TruncatedPowerLaw(alpha=2.35, low=0.1, high=10.0), (0.0, 0.1, 0.9, 1.0)),
        (TruncatedPowerLaw(alpha=1.0, low=0.1, high=10.0), (0.0, 0.1, 0.9, 1.0)),
        (TruncatedSine(), (0.0, 0.1, 0.5, 0.9, 1.0)),
    ],
)
def test_continuous_cdf_ppf_round_trip(distribution, probabilities) -> None:
    for probability in probabilities:
        assert distribution.cdf(distribution.ppf(probability)) == pytest.approx(
            probability, rel=2e-9, abs=2e-11
        )


def test_normal_matches_definition() -> None:
    distribution = Normal(loc=1.0, scale=2.0)
    expected = -0.5 * ((3.0 - 1.0) / 2.0) ** 2
    expected -= math.log(2.0) + 0.5 * math.log(2.0 * math.pi)
    assert distribution.logpdf(3.0) == pytest.approx(expected)
    assert distribution.pdf(3.0) == pytest.approx(math.exp(expected))
    assert distribution.mean == 1.0
    assert distribution.median == 1.0


@pytest.mark.parametrize(
    ("distribution", "below", "above"),
    [
        (Uniform(loc=1.0, scale=2.0), 0.99, 3.01),
        (Beta(a=2.0, b=3.0, loc=1.0, scale=2.0), 0.99, 3.01),
        (Exponential(), -0.01, math.inf),
        (TruncatedNormal(0.0, 1.0, -1.0, 1.0), -1.01, 1.01),
        (TruncatedPowerLaw(2.0, 1.0, 4.0), 0.99, 4.01),
        (TruncatedSine(), -0.01, math.pi / 2.0 + 0.01),
    ],
)
def test_density_is_zero_outside_finite_support(distribution, below, above) -> None:
    assert distribution.pdf(below) == 0.0
    if math.isfinite(above):
        assert distribution.pdf(above) == 0.0


def test_beta_known_cases_and_boundaries() -> None:
    uniform_beta = Beta()
    assert uniform_beta.pdf(0.0) == pytest.approx(1.0)
    assert uniform_beta.pdf(1.0) == pytest.approx(1.0)
    assert uniform_beta.cdf(0.25) == pytest.approx(0.25)

    symmetric = Beta(a=2.0, b=2.0)
    assert symmetric.pdf(0.5) == pytest.approx(1.5)
    assert symmetric.cdf(0.5) == pytest.approx(0.5)
    assert symmetric.ppf(0.5) == pytest.approx(0.5)

    singular = Beta(a=0.5, b=2.0)
    assert singular.pdf(0.0) == math.inf
    assert singular.pdf(1.0) == 0.0


def test_truncated_normal_is_renormalized() -> None:
    distribution = TruncatedNormal(loc=0.0, scale=1.0, low=-1.0, high=1.0)
    untruncated_at_zero = 1.0 / math.sqrt(2.0 * math.pi)
    retained_mass = Normal().cdf(1.0) - Normal().cdf(-1.0)
    assert distribution.pdf(0.0) == pytest.approx(
        untruncated_at_zero / retained_mass
    )


def test_discrete_uniform_corrects_pbjam_logpdf_bug() -> None:
    distribution = DiscreteUniform(low=2, high=6)
    assert distribution.pdf(2) == pytest.approx(0.25)
    assert distribution.pdf(2.5) == 0.0
    assert distribution.logpdf(2) == pytest.approx(-math.log(4.0))
    assert distribution.logpdf(6) == -math.inf
    assert distribution.cdf(1.9) == 0.0
    assert distribution.cdf(2.0) == pytest.approx(0.25)
    assert distribution.cdf(5.0) == 1.0
    assert [distribution.ppf(q) for q in (0.0, 0.25, 0.5, 0.75, 1.0)] == [
        2,
        2,
        3,
        4,
        5,
    ]


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: Normal(scale=0.0),
        lambda: Uniform(scale=-1.0),
        lambda: Beta(a=0.0),
        lambda: Beta(b=-1.0),
        lambda: Exponential(scale=math.inf),
        lambda: TruncatedNormal(0.0, 1.0, 2.0, 1.0),
        lambda: TruncatedPowerLaw(2.0, 0.0, 1.0),
        lambda: DiscreteUniform(3, 3),
    ],
)
def test_invalid_parameters_fail_at_construction(constructor) -> None:
    with pytest.raises(ValueError):
        constructor()


@pytest.mark.parametrize(
    "distribution",
    [
        Normal(),
        Uniform(),
        Beta(),
        Exponential(),
        TruncatedNormal(0.0, 1.0, -1.0, 1.0),
        TruncatedPowerLaw(1.0, 0.1, 10.0),
        TruncatedSine(),
    ],
)
def test_invalid_quantiles_return_nan(distribution) -> None:
    assert math.isnan(distribution.ppf(-0.1))
    assert math.isnan(distribution.ppf(1.1))


def test_pbjam_lower_case_names_remain_available() -> None:
    assert normal is Normal
    assert uniform is Uniform
    assert beta is Beta
    assert truncsine is TruncatedSine
    assert randint is DiscreteUniform
