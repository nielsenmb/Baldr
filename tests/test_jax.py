"""Numerical and transformation tests for the optional JAX backend."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import baldr
from baldr import jax as baldr_jax

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    ("distribution", "reference", "points"),
    [
        (baldr_jax.Normal(1.0, 2.0), stats.norm(1.0, 2.0), [-4.0, 0.0, 1.0, 5.0]),
        (
            baldr_jax.Uniform(-1.0, 3.0),
            stats.uniform(-1.0, 3.0),
            [-2.0, -1.0, 0.0, 2.0],
        ),
        (
            baldr_jax.Beta(0.7, 2.5, -1.0, 3.0),
            stats.beta(0.7, 2.5, -1.0, 3.0),
            [-1.0, -0.5, 0.0, 2.0],
        ),
        (
            baldr_jax.Exponential(2.0),
            stats.expon(scale=2.0),
            [-1.0, 0.0, 1.0, 5.0],
        ),
        (
            baldr_jax.TruncatedNormal(0.0, 1.5, -1.0, 2.0),
            stats.truncnorm(-1.0 / 1.5, 2.0 / 1.5, 0.0, 1.5),
            [-2.0, -1.0, 0.0, 2.0],
        ),
    ],
)
def test_continuous_distributions_match_scipy(distribution, reference, points):
    points = jnp.asarray(points)
    np.testing.assert_allclose(distribution.pdf(points), reference.pdf(points))
    np.testing.assert_allclose(distribution.cdf(points), reference.cdf(points))
    probabilities = jnp.asarray([0.01, 0.2, 0.5, 0.8, 0.99])
    np.testing.assert_allclose(
        distribution.ppf(probabilities),
        reference.ppf(probabilities),
        rtol=2e-10,
        atol=2e-10,
    )


@pytest.mark.parametrize(
    "distribution",
    [
        baldr_jax.Normal(),
        baldr_jax.Uniform(),
        baldr_jax.Beta(2.0, 3.0),
        baldr_jax.Exponential(),
        baldr_jax.TruncatedNormal(0.0, 1.0, -1.0, 1.0),
        baldr_jax.TruncatedPowerLaw(2.35, 0.1, 10.0),
        baldr_jax.TruncatedPowerLaw(1.0, 0.1, 10.0),
        baldr_jax.TruncatedSine(),
    ],
)
def test_cdf_ppf_round_trip_under_jit(distribution):
    probabilities = jnp.asarray([[0.01, 0.5], [0.9, 0.99]])
    transform = jax.jit(distribution.ppf)
    result = transform(probabilities)
    assert result.shape == probabilities.shape
    np.testing.assert_allclose(
        distribution.cdf(result), probabilities, rtol=2e-10, atol=2e-10
    )


def test_public_api_selects_jax_backend_lazily():
    distribution = baldr.Normal(backend="jax")
    assert isinstance(distribution, baldr_jax.Normal)
    assert isinstance(distribution.pdf(0.0), jax.Array)


def test_vmap_and_gradient_are_supported():
    distribution = baldr_jax.Normal(loc=0.5, scale=2.0)
    points = jnp.asarray([-1.0, 0.0, 1.0])
    values = jax.vmap(distribution.logpdf)(points)
    np.testing.assert_allclose(values, distribution.logpdf(points))
    assert jax.grad(distribution.logpdf)(0.5) == pytest.approx(0.0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_float_width_is_preserved(dtype):
    result = jax.jit(baldr_jax.Normal().ppf)(
        jnp.asarray([0.2, 0.8], dtype=dtype)
    )
    assert result.dtype == dtype


def test_discrete_uniform_and_beta_boundaries():
    discrete = baldr_jax.DiscreteUniform(2, 6)
    np.testing.assert_allclose(
        discrete.pdf(jnp.asarray([1.0, 2.0, 2.5, 5.0, 6.0])),
        [0.0, 0.25, 0.0, 0.25, 0.0],
    )
    beta = baldr_jax.Beta(0.5, 2.0)
    assert beta.pdf(0.0) == math.inf
    assert beta.pdf(1.0) == 0.0
