"""Tests for the dependency-free scalar backend."""

import math

import pytest
from scipy import stats

from baldr.scalar import (
    Beta,
    Cauchy,
    DiscreteUniform,
    Exponential,
    Gamma,
    HalfNormal,
    Laplace,
    LogNormal,
    Normal,
    StudentT,
    TruncatedNormal,
    TruncatedPowerLaw,
    TruncatedSine,
    Uniform,
    Weibull,
    beta,
    gamma,
    normal,
    randint,
    truncsine,
    uniform,
)


@pytest.mark.parametrize(
    ("distribution", "reference"),
    [
        (LogNormal(0.7, -1.0, 2.5), -1.0),
        (HalfNormal(-1.0, 2.5), -1.0),
        (Cauchy(1.0, 2.5), -math.inf),
        (Laplace(1.0, 2.5), -math.inf),
        (Weibull(1.7, -1.0, 2.5), -1.0),
    ],
)
def test_additional_scalar_distributions_round_trip(distribution, reference):
    """New scalar distributions invert their CDFs and preserve endpoints."""

    assert distribution.ppf(0.0) == reference
    for probability in (1e-6, 0.1, 0.5, 0.9, 1.0 - 1e-6):
        assert distribution.cdf(distribution.ppf(probability)) == pytest.approx(
            probability, rel=2e-9, abs=2e-11
        )


def test_scalar_normal_logsf_is_stable_in_extreme_tail():
    """Scalar Normal log-survival avoids CDF subtraction and underflow."""

    distribution = Normal()
    assert distribution.cdf(40.0) == 1.0
    assert distribution.logsf(40.0) == pytest.approx(-804.6084420137539, rel=1e-8)

    gamma_distribution = Gamma(a=2.5)
    assert gamma_distribution.cdf(100.0) == 1.0
    assert gamma_distribution.logsf(100.0) == pytest.approx(
        stats.gamma(2.5).logsf(100.0), rel=2e-12
    )


@pytest.mark.parametrize(
    ("distribution", "reference", "points"),
    [
        (LogNormal(0.7, -1.0, 2.5), stats.lognorm(0.7, -1.0, 2.5), (-1.0, 0.0, 5.0)),
        (HalfNormal(-1.0, 2.5), stats.halfnorm(-1.0, 2.5), (-2.0, -1.0, 5.0)),
        (Cauchy(1.0, 2.5), stats.cauchy(1.0, 2.5), (-20.0, 1.0, 20.0)),
        (Laplace(1.0, 2.5), stats.laplace(1.0, 2.5), (-20.0, 1.0, 20.0)),
        (
            Weibull(1.7, -1.0, 2.5),
            stats.weibull_min(1.7, -1.0, 2.5),
            (-2.0, -1.0, 5.0),
        ),
    ],
)
def test_additional_scalar_distributions_match_scipy(distribution, reference, points):
    """Scalar density and tail methods agree with SciPy."""

    for point in points:
        for method in ("logpdf", "pdf", "cdf", "sf", "logcdf", "logsf"):
            assert getattr(distribution, method)(point) == pytest.approx(
                getattr(reference, method)(point), rel=2e-10, abs=2e-12
            )


@pytest.mark.parametrize(
    ("distribution", "probabilities"),
    [
        (Normal(loc=2.0, scale=3.0), (1e-10, 0.01, 0.5, 0.99, 1 - 1e-10)),
        (Uniform(loc=-2.0, scale=5.0), (0.0, 0.1, 0.5, 0.9, 1.0)),
        (Beta(a=0.4, b=3.2, loc=-1.0, scale=4.0), (1e-4, 0.01, 0.5, 0.99)),
        (Exponential(scale=2.5), (0.0, 0.1, 0.5, 0.99)),
        (Gamma(a=2.5, loc=-1.0, scale=3.0), (0.0, 0.01, 0.5, 0.99)),
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
        (Gamma(a=2.0, loc=1.0), 0.99, math.inf),
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


@pytest.mark.parametrize(
    ("a", "b"),
    [(0.01, 1.0), (0.1, 10.0), (0.5, 0.5), (10.0, 0.1), (100.0, 100.0)],
)
def test_scalar_beta_inverse_cdf_hard_grid(a, b) -> None:
    """Safeguarded scalar Beta inversion agrees with SciPy across shapes."""

    distribution = Beta(a, b)
    reference = stats.beta(a, b)
    for probability in (1e-12, 1e-6, 1e-3, 0.1, 0.5, 0.9, 1.0 - 1e-9):
        expected = reference.ppf(probability)
        actual = distribution.ppf(probability)
        assert actual == pytest.approx(expected, rel=2e-8, abs=2e-13)


def test_gamma_known_cases_and_boundaries() -> None:
    distribution = Gamma(a=2.0, loc=1.0, scale=3.0)
    assert distribution.pdf(1.0) == 0.0
    assert distribution.pdf(4.0) == pytest.approx(math.exp(-1.0) / 3.0)
    assert distribution.cdf(4.0) == pytest.approx(1.0 - 2.0 / math.e)
    assert distribution.mean == 7.0


@pytest.mark.parametrize("df", [0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
def test_student_t_matches_scipy(df) -> None:
    """Scalar Student's t methods agree with SciPy across tail weights."""

    distribution = StudentT(df=df, loc=-1.0, scale=2.5)
    reference = stats.t(df=df, loc=-1.0, scale=2.5)
    for point in (-100.0, -3.0, -1.0, 2.0, 100.0):
        assert distribution.logpdf(point) == pytest.approx(reference.logpdf(point))
        assert distribution.cdf(point) == pytest.approx(reference.cdf(point))
        assert distribution.sf(point) == pytest.approx(reference.sf(point))
        assert distribution.logcdf(point) == pytest.approx(reference.logcdf(point))
        assert distribution.logsf(point) == pytest.approx(reference.logsf(point))
    for probability in (0.0, 1e-9, 0.01, 0.5, 0.99, 1.0 - 1e-9, 1.0):
        assert distribution.ppf(probability) == pytest.approx(
            reference.ppf(probability), rel=2e-8, abs=2e-12
        )


@pytest.mark.parametrize("shape", [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
def test_scalar_gamma_inverse_cdf_hard_grid(shape) -> None:
    """Safeguarded scalar Gamma inversion agrees with SciPy across shapes."""

    distribution = Gamma(shape)
    reference = stats.gamma(shape)
    for probability in (1e-12, 1e-6, 1e-3, 0.1, 0.5, 0.9, 1.0 - 1e-9):
        assert distribution.ppf(probability) == pytest.approx(
            reference.ppf(probability), rel=2e-8, abs=2e-13
        )


def test_truncated_normal_is_renormalized() -> None:
    distribution = TruncatedNormal(loc=0.0, scale=1.0, low=-1.0, high=1.0)
    untruncated_at_zero = 1.0 / math.sqrt(2.0 * math.pi)
    retained_mass = Normal().cdf(1.0) - Normal().cdf(-1.0)
    assert distribution.pdf(0.0) == pytest.approx(untruncated_at_zero / retained_mass)


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
        lambda: Gamma(a=0.0),
        lambda: StudentT(df=0.0),
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
        Gamma(),
        StudentT(),
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
    assert gamma is Gamma
    assert truncsine is TruncatedSine
    assert randint is DiscreteUniform
