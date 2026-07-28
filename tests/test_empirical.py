"""Tests for fitted empirical distributions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baldr.empirical import EmpiricalGrid, fit_empirical, fit_empirical_marginals


@pytest.fixture
def sample():
    """Return a deterministic non-Gaussian sample."""

    rng = np.random.default_rng(42)
    return np.concatenate((rng.normal(-1.0, 0.4, 300), rng.normal(1.2, 0.7, 700)))


def test_numpy_empirical_is_normalized_and_broadcasts(sample):
    """Check numerical normalization, shapes, and support behaviour."""

    distribution = fit_empirical(sample, grid_size=1024)
    grid = distribution.grid

    density = distribution.pdf(grid.support)
    mass = np.sum(0.5 * (density[1:] + density[:-1]) * np.diff(grid.support))
    assert mass == pytest.approx(1.0)
    outside = [grid.support[0] - 1.0, grid.support[-1] + 1.0]
    assert distribution.pdf(outside).tolist() == [0.0, 0.0]
    assert distribution.cdf(outside).tolist() == [0.0, 1.0]
    assert distribution.ppf(np.array([[0.1, 0.5], [0.9, 0.99]])).shape == (2, 2)


def test_empirical_cdf_ppf_round_trip(sample):
    """Check interpolation consistency over central probabilities."""

    distribution = fit_empirical(sample, grid_size=2048)
    probabilities = np.linspace(0.01, 0.99, 99)

    np.testing.assert_allclose(
        distribution.cdf(distribution.ppf(probabilities)),
        probabilities,
        atol=2e-6,
    )
    assert np.isnan(distribution.ppf([-0.1, 1.1])).all()


def test_jax_backend_reuses_grid_and_is_traceable(sample):
    """Check whole-function JIT without repeating the KDE fit."""

    numpy_distribution = fit_empirical(sample)
    jax_distribution = numpy_distribution.grid.to_backend("jax")
    probabilities = jnp.asarray([0.1, 0.5, 0.9])
    transform = jax.jit(jax_distribution.ppf)
    quantiles = transform(probabilities)

    np.testing.assert_allclose(
        np.asarray(quantiles),
        numpy_distribution.ppf(np.asarray(probabilities)),
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(jax_distribution.cdf(quantiles)),
        np.asarray(probabilities),
        atol=2e-6,
    )


def test_fit_empirical_marginals_preserves_column_order(sample):
    """Fit independent columns using the PBjam-style data layout."""

    distributions = fit_empirical_marginals(np.column_stack((sample, 2.0 * sample)))

    assert len(distributions) == 2
    assert distributions[1].median == pytest.approx(
        2.0 * distributions[0].median, rel=2e-3
    )


@pytest.mark.parametrize(
    "sample",
    [
        [1.0],
        [1.0, 1.0],
        [1.0, np.nan],
        [[1.0, 2.0], [3.0, 4.0]],
    ],
)
def test_invalid_samples_fail_at_fit(sample):
    """Reject data that cannot define a one-dimensional KDE."""

    with pytest.raises(ValueError, match="sample"):
        fit_empirical(sample)


def test_grid_rejects_inconsistent_arrays():
    """Reject malformed precomputed interpolation grids."""

    with pytest.raises(ValueError):
        EmpiricalGrid(
            support=np.arange(4.0),
            density=np.ones(4),
            cumulative=np.asarray([0.0, 0.5, 0.4, 1.0]),
        )


def test_unknown_empirical_backend_fails(sample):
    """Reject unsupported evaluation backends at construction."""

    with pytest.raises(ValueError, match="backend"):
        fit_empirical(sample, backend="other")
