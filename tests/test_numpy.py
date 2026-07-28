"""Tests for the broadcasting NumPy backend."""

import math

import numpy as np
import pytest
from scipy import stats

from baldr.numpy import (
    Beta,
    DiscreteUniform,
    Exponential,
    Normal,
    TruncatedNormal,
    TruncatedPowerLaw,
    TruncatedSine,
    Uniform,
)


@pytest.mark.parametrize(
    ("distribution", "reference", "points"),
    [
        (Normal(1.0, 2.0), stats.norm(1.0, 2.0), [-4.0, 0.0, 1.0, 5.0]),
        (Uniform(-1.0, 3.0), stats.uniform(-1.0, 3.0), [-2.0, -1.0, 0.0, 2.0]),
        (
            Beta(0.7, 2.5, -1.0, 3.0),
            stats.beta(0.7, 2.5, -1.0, 3.0),
            [-1.0, -0.5, 0.0, 2.0],
        ),
        (Exponential(2.0), stats.expon(scale=2.0), [-1.0, 0.0, 1.0, 5.0]),
        (
            TruncatedNormal(0.0, 1.5, -1.0, 2.0),
            stats.truncnorm(-1.0 / 1.5, 2.0 / 1.5, 0.0, 1.5),
            [-2.0, -1.0, 0.0, 2.0],
        ),
    ],
)
def test_continuous_distributions_match_scipy(distribution, reference, points):
    points = np.asarray(points)
    np.testing.assert_allclose(distribution.pdf(points), reference.pdf(points))
    np.testing.assert_allclose(distribution.cdf(points), reference.cdf(points))
    probabilities = np.asarray([0.0, 0.01, 0.5, 0.99, 1.0])
    np.testing.assert_allclose(
        distribution.ppf(probabilities), reference.ppf(probabilities)
    )


@pytest.mark.parametrize(
    "distribution",
    [
        Normal(),
        Uniform(),
        Beta(2.0, 3.0),
        Exponential(),
        TruncatedNormal(0.0, 1.0, -1.0, 1.0),
        TruncatedPowerLaw(2.35, 0.1, 10.0),
        TruncatedPowerLaw(1.0, 0.1, 10.0),
        TruncatedSine(),
    ],
)
def test_cdf_ppf_round_trip_and_shape(distribution):
    probabilities = np.asarray([[0.01, 0.5], [0.9, 0.99]])
    result = distribution.ppf(probabilities)
    assert result.shape == probabilities.shape
    np.testing.assert_allclose(distribution.cdf(result), probabilities, rtol=1e-11)


def test_methods_accept_lists_and_scalar_values():
    distribution = Normal()
    assert distribution.pdf([0.0, 1.0]).shape == (2,)
    assert np.ndim(distribution.pdf(0.0)) == 0


def test_discrete_uniform_broadcasts_and_handles_endpoints():
    distribution = DiscreteUniform(2, 6)
    np.testing.assert_allclose(
        distribution.pdf([1.0, 2.0, 2.5, 5.0, 6.0]),
        [0.0, 0.25, 0.0, 0.25, 0.0],
    )
    np.testing.assert_allclose(
        distribution.ppf([0.0, 0.25, 0.5, 0.75, 1.0]),
        [2.0, 2.0, 3.0, 4.0, 5.0],
    )


def test_invalid_quantiles_return_nan():
    for distribution in (Normal(), Uniform(), Beta(), Exponential()):
        assert np.isnan(distribution.ppf([-0.1, 1.1])).all()


def test_beta_boundary_behaviour():
    regular = Beta()
    np.testing.assert_allclose(regular.pdf([0.0, 1.0]), [1.0, 1.0])
    singular = Beta(0.5, 2.0)
    assert singular.pdf(0.0) == math.inf
    assert singular.pdf(1.0) == 0.0
