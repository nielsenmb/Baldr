"""Interpolated empirical distributions fitted from one-dimensional samples.

Fitting uses NumPy and SciPy once to construct normalized interpolation grids.
Evaluation then uses either NumPy or JAX interpolation without re-evaluating
the kernel density estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.stats import gaussian_kde

EmpiricalBackend = Literal["numpy", "jax"]

__all__ = [
    "EmpiricalBackend",
    "EmpiricalGrid",
    "JAXEmpiricalDistribution",
    "NumPyEmpiricalDistribution",
    "fit_empirical",
    "fit_empirical_marginals",
]


def _normal_reference_factor(sample: np.ndarray) -> float:
    """Calculate the robust normal-reference KDE bandwidth factor.

    Parameters
    ----------
    sample : numpy.ndarray
        Finite one-dimensional sample.

    Returns
    -------
    float
        Bandwidth expressed as a multiple of the sample standard deviation.
    """

    standard_deviation = float(np.std(sample, ddof=1))
    quartiles = np.percentile(sample, [25.0, 75.0])
    robust_scale = min(standard_deviation, float(np.diff(quartiles)[0]) / 1.349)
    if robust_scale <= 0.0:
        raise ValueError("sample must have non-zero robust spread")
    bandwidth = 1.0592238410488122 * robust_scale * sample.size ** (-0.2)
    return bandwidth / standard_deviation


def _validate_sample(sample: Any) -> np.ndarray:
    """Return a validated one-dimensional floating-point sample.

    Parameters
    ----------
    sample : array-like
        Values used to fit an empirical distribution.

    Returns
    -------
    numpy.ndarray
        Contiguous one-dimensional sample.
    """

    values = np.asarray(sample, dtype=float)
    if values.ndim != 1:
        raise ValueError("sample must be one-dimensional")
    if values.size < 2:
        raise ValueError("sample must contain at least two values")
    if not np.all(np.isfinite(values)):
        raise ValueError("sample must contain only finite values")
    if float(np.std(values, ddof=1)) == 0.0:
        raise ValueError("sample must have non-zero spread")
    return np.ascontiguousarray(values)


def _readonly(values: Any) -> np.ndarray:
    """Return a read-only one-dimensional floating-point array.

    Parameters
    ----------
    values : array-like
        Values to copy and protect from mutation.

    Returns
    -------
    numpy.ndarray
        Read-only array.
    """

    array = np.array(values, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _trapezoid(values: np.ndarray, support: np.ndarray) -> float:
    """Integrate tabulated values without requiring a recent NumPy version.

    Parameters
    ----------
    values : numpy.ndarray
        Function values on the integration grid.
    support : numpy.ndarray
        Strictly increasing integration grid.

    Returns
    -------
    float
        Trapezoidal integral.
    """

    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(support)
    return float(np.sum(increments))


@dataclass(frozen=True, slots=True)
class EmpiricalGrid:
    """Store a fitted density and cumulative distribution on a fixed grid.

    Parameters
    ----------
    support : array-like
        Strictly increasing evaluation grid.
    density : array-like
        Non-negative normalized density values.
    cumulative : array-like
        Non-decreasing cumulative probabilities with endpoints zero and one.
    """

    support: np.ndarray
    density: np.ndarray
    cumulative: np.ndarray
    _ppf_probability: np.ndarray = field(init=False, repr=False)
    _ppf_value: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate fitted arrays and cache a strictly increasing PPF grid."""

        support = _readonly(self.support)
        density = _readonly(self.density)
        cumulative = _readonly(self.cumulative)
        if support.ndim != 1 or support.size < 4:
            message = "support must be one-dimensional with at least four points"
            raise ValueError(message)
        if density.shape != support.shape or cumulative.shape != support.shape:
            message = "support, density, and cumulative must have matching shapes"
            raise ValueError(message)
        if not np.all(np.isfinite(support)) or not np.all(np.diff(support) > 0.0):
            raise ValueError("support must be finite and strictly increasing")
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("density must be finite and non-negative")
        if not np.all(np.isfinite(cumulative)) or np.any(np.diff(cumulative) < 0.0):
            raise ValueError("cumulative must be finite and non-decreasing")
        if cumulative[0] != 0.0 or cumulative[-1] != 1.0:
            raise ValueError("cumulative endpoints must be zero and one")

        unique = np.concatenate(([True], np.diff(cumulative) > 0.0))
        unique[-1] = True
        ppf_probability = _readonly(cumulative[unique])
        ppf_value = _readonly(support[unique])
        if ppf_probability.size < 2:
            raise ValueError("cumulative must contain increasing probabilities")

        object.__setattr__(self, "support", support)
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "cumulative", cumulative)
        object.__setattr__(self, "_ppf_probability", ppf_probability)
        object.__setattr__(self, "_ppf_value", ppf_value)

    @property
    def mean(self) -> float:
        """Return the mean of the interpolated density."""

        return _trapezoid(self.support * self.density, self.support)

    def to_backend(
        self, backend: EmpiricalBackend = "numpy"
    ) -> NumPyEmpiricalDistribution | JAXEmpiricalDistribution:
        """Construct an evaluator without refitting the KDE.

        Parameters
        ----------
        backend : {"numpy", "jax"}, default="numpy"
            Array implementation used for repeated evaluations.

        Returns
        -------
        NumPyEmpiricalDistribution or JAXEmpiricalDistribution
            Backend-specific evaluator sharing the fitted grid values.
        """

        if backend == "numpy":
            return NumPyEmpiricalDistribution(self)
        if backend == "jax":
            return JAXEmpiricalDistribution(self)
        raise ValueError("backend must be 'numpy' or 'jax'")


@dataclass(frozen=True, slots=True)
class NumPyEmpiricalDistribution:
    """Evaluate a fitted empirical grid with broadcasting NumPy operations.

    Parameters
    ----------
    grid : EmpiricalGrid
        Precomputed interpolation grid.
    """

    grid: EmpiricalGrid

    @property
    def mean(self) -> float:
        """Return the interpolated distribution mean."""

        return self.grid.mean

    @property
    def median(self) -> float:
        """Return the interpolated distribution median."""

        return float(self.ppf(0.5))

    def pdf(self, x: Any) -> np.ndarray:
        """Evaluate the probability density.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        numpy.ndarray
            Interpolated density, with zero outside the fitted support.
        """

        return np.interp(
            np.asarray(x), self.grid.support, self.grid.density, left=0.0, right=0.0
        )

    def logpdf(self, x: Any) -> np.ndarray:
        """Evaluate the log-probability density.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        numpy.ndarray
            Interpolated log-density, with negative infinity outside support.
        """

        with np.errstate(divide="ignore"):
            return np.log(self.pdf(x))

    def cdf(self, x: Any) -> np.ndarray:
        """Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        numpy.ndarray
            Interpolated cumulative probabilities.
        """

        return np.interp(
            np.asarray(x),
            self.grid.support,
            self.grid.cumulative,
            left=0.0,
            right=1.0,
        )

    def ppf(self, q: Any) -> np.ndarray:
        """Evaluate the percent-point function.

        Parameters
        ----------
        q : array-like
            Cumulative probabilities in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray
            Interpolated quantiles; invalid probabilities return NaN.
        """

        probability = np.asarray(q)
        value = np.interp(
            probability, self.grid._ppf_probability, self.grid._ppf_value
        )
        valid = (probability >= 0.0) & (probability <= 1.0)
        return np.where(valid, value, np.nan)


@dataclass(frozen=True, slots=True)
class JAXEmpiricalDistribution:
    """Evaluate a fitted empirical grid with JAX-traceable interpolation.

    Parameters
    ----------
    grid : EmpiricalGrid
        Precomputed interpolation grid. Importing and constructing this class
        requires the optional JAX dependency.
    """

    grid: EmpiricalGrid
    _support: Any = field(init=False, repr=False)
    _density: Any = field(init=False, repr=False)
    _cumulative: Any = field(init=False, repr=False)
    _ppf_probability: Any = field(init=False, repr=False)
    _ppf_value: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Transfer the fitted interpolation grids to JAX arrays once."""

        import jax.numpy as jnp

        object.__setattr__(self, "_support", jnp.asarray(self.grid.support))
        object.__setattr__(self, "_density", jnp.asarray(self.grid.density))
        object.__setattr__(self, "_cumulative", jnp.asarray(self.grid.cumulative))
        object.__setattr__(
            self, "_ppf_probability", jnp.asarray(self.grid._ppf_probability)
        )
        object.__setattr__(self, "_ppf_value", jnp.asarray(self.grid._ppf_value))

    @property
    def mean(self) -> float:
        """Return the interpolated distribution mean."""

        return self.grid.mean

    @property
    def median(self) -> Any:
        """Return the interpolated distribution median."""

        return self.ppf(0.5)

    def pdf(self, x: Any) -> Any:
        """Evaluate the probability density.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        jax.Array
            Interpolated density, with zero outside the fitted support.
        """

        import jax.numpy as jnp

        return jnp.interp(
            jnp.asarray(x), self._support, self._density, left=0.0, right=0.0
        )

    def logpdf(self, x: Any) -> Any:
        """Evaluate the log-probability density.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        jax.Array
            Interpolated log-density, with negative infinity outside support.
        """

        import jax.numpy as jnp

        return jnp.log(self.pdf(x))

    def cdf(self, x: Any) -> Any:
        """Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        jax.Array
            Interpolated cumulative probabilities.
        """

        import jax.numpy as jnp

        return jnp.interp(
            jnp.asarray(x),
            self._support,
            self._cumulative,
            left=0.0,
            right=1.0,
        )

    def ppf(self, q: Any) -> Any:
        """Evaluate the percent-point function.

        Parameters
        ----------
        q : array-like
            Cumulative probabilities in ``[0, 1]``.

        Returns
        -------
        jax.Array
            Interpolated quantiles; invalid probabilities return NaN.
        """

        import jax.numpy as jnp

        probability = jnp.asarray(q)
        value = jnp.interp(probability, self._ppf_probability, self._ppf_value)
        valid = (probability >= 0.0) & (probability <= 1.0)
        return jnp.where(valid, value, jnp.nan)


def fit_empirical(
    sample: Any,
    *,
    backend: EmpiricalBackend = "numpy",
    bandwidth: str | float = "normal_reference",
    cut: float = 5.0,
    grid_size: int = 512,
) -> NumPyEmpiricalDistribution | JAXEmpiricalDistribution:
    """Fit a KDE once and construct an interpolated empirical distribution.

    Parameters
    ----------
    sample : array-like
        Finite one-dimensional sample.
    backend : {"numpy", "jax"}, default="numpy"
        Array implementation used for repeated evaluations.
    bandwidth : {"normal_reference", "scott", "silverman"} or float, \
            default="normal_reference"
        KDE bandwidth rule or positive SciPy ``bw_method`` factor. The default
        reproduces PBjam's robust normal-reference rule.
    cut : float, default=5.0
        Number of fitted kernel bandwidths added beyond each sample extreme.
    grid_size : int, default=512
        Number of support points used for cached interpolation.

    Returns
    -------
    NumPyEmpiricalDistribution or JAXEmpiricalDistribution
        Backend-specific evaluator.
    """

    values = _validate_sample(sample)
    if not isinstance(grid_size, int) or isinstance(grid_size, bool) or grid_size < 16:
        raise ValueError("grid_size must be an integer of at least 16")
    cut = float(cut)
    if not math.isfinite(cut) or cut <= 0.0:
        raise ValueError("cut must be finite and greater than zero")

    if bandwidth == "normal_reference":
        bandwidth_method: str | float = _normal_reference_factor(values)
    elif isinstance(bandwidth, str):
        if bandwidth not in {"scott", "silverman"}:
            raise ValueError(
                "bandwidth must be 'normal_reference', 'scott', 'silverman', "
                "or a positive float"
            )
        bandwidth_method = bandwidth
    else:
        bandwidth_method = float(bandwidth)
        if not math.isfinite(bandwidth_method) or bandwidth_method <= 0.0:
            raise ValueError("numeric bandwidth must be finite and greater than zero")

    kde = gaussian_kde(values, bw_method=bandwidth_method)
    kernel_bandwidth = math.sqrt(float(kde.covariance[0, 0]))
    support = np.linspace(
        float(np.min(values)) - cut * kernel_bandwidth,
        float(np.max(values)) + cut * kernel_bandwidth,
        grid_size,
    )
    density = np.asarray(kde(support))
    mass = _trapezoid(density, support)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("fitted KDE has no finite probability mass")
    density /= mass

    increments = 0.5 * (density[1:] + density[:-1]) * np.diff(support)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    cumulative /= cumulative[-1]
    cumulative[0] = 0.0
    cumulative[-1] = 1.0
    grid = EmpiricalGrid(support, density, cumulative)
    return grid.to_backend(backend)


def fit_empirical_marginals(
    samples: Any,
    *,
    backend: EmpiricalBackend = "numpy",
    bandwidth: str | float = "normal_reference",
    cut: float = 5.0,
    grid_size: int = 512,
) -> tuple[NumPyEmpiricalDistribution | JAXEmpiricalDistribution, ...]:
    """Fit one independent empirical distribution per sample column.

    Parameters
    ----------
    samples : array-like
        Two-dimensional array shaped ``(n_samples, n_dimensions)``.
    backend : {"numpy", "jax"}, default="numpy"
        Array implementation used for repeated evaluations.
    bandwidth : {"normal_reference", "scott", "silverman"} or float, \
            default="normal_reference"
        KDE bandwidth rule passed to :func:`fit_empirical`.
    cut : float, default=5.0
        Number of fitted kernel bandwidths added beyond each sample extreme.
    grid_size : int, default=512
        Number of support points used for each cached interpolation.

    Returns
    -------
    tuple
        Independent marginal empirical distributions in column order.
    """

    values = np.asarray(samples)
    if values.ndim != 2:
        raise ValueError("samples must have shape (n_samples, n_dimensions)")
    return tuple(
        fit_empirical(
            values[:, index],
            backend=backend,
            bandwidth=bandwidth,
            cut=cut,
            grid_size=grid_size,
        )
        for index in range(values.shape[1])
    )
