"""Backend-selecting public distribution constructors."""

from __future__ import annotations

from typing import Any, Literal

from baldr import scalar

Backend = Literal["scalar", "numpy", "jax"]

__all__ = [
    "Backend",
    "Beta",
    "Cauchy",
    "DiscreteUniform",
    "Exponential",
    "Gamma",
    "HalfNormal",
    "Laplace",
    "LogNormal",
    "Normal",
    "TruncatedNormal",
    "TruncatedPowerLaw",
    "TruncatedSine",
    "Uniform",
    "Weibull",
]


def _module(backend: Backend) -> Any:
    """Load an execution backend lazily.

    Parameters
    ----------
    backend : {"scalar", "numpy", "jax"}
        Requested execution backend.

    Returns
    -------
    module
        Backend implementation module.
    """

    if backend == "scalar":
        return scalar
    if backend == "numpy":
        from baldr import numpy

        return numpy
    if backend == "jax":
        from baldr import jax

        return jax
    raise ValueError("backend must be 'scalar', 'numpy', or 'jax'")


def Normal(loc: float = 0.0, scale: float = 1.0, *, backend: Backend = "scalar") -> Any:
    """Construct a Normal distribution for the selected backend.

    Parameters
    ----------
    loc : float, default=0.0
        Distribution mean.
    scale : float, default=1.0
        Positive standard deviation.
    backend : {"scalar", "numpy", "jax"}, default="scalar"
        Execution backend selected once during construction.

    Returns
    -------
    object
        Normal distribution implemented by the selected backend.
    """

    return _module(backend).Normal(loc=loc, scale=scale)


def Uniform(
    loc: float = 0.0, scale: float = 1.0, *, backend: Backend = "scalar"
) -> Any:
    """Construct a continuous uniform distribution for the selected backend.

    Parameters
    ----------
    loc : float, default=0.0
        Lower support boundary.
    scale : float, default=1.0
        Positive support width.
    backend : {"scalar", "numpy", "jax"}, default="scalar"
        Execution backend selected once during construction.

    Returns
    -------
    object
        Uniform distribution implemented by the selected backend.
    """

    return _module(backend).Uniform(loc=loc, scale=scale)


def Beta(
    a: float = 1.0,
    b: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    *,
    backend: Backend = "scalar",
) -> Any:
    """Construct a transformed Beta distribution for the selected backend.

    Parameters
    ----------
    a, b : float, default=1.0
        Positive shape parameters.
    loc : float, default=0.0
        Lower support boundary.
    scale : float, default=1.0
        Positive support width.
    backend : {"scalar", "numpy", "jax"}, default="scalar"
        Execution backend selected once during construction.

    Returns
    -------
    object
        Beta distribution implemented by the selected backend.
    """

    return _module(backend).Beta(a=a, b=b, loc=loc, scale=scale)


def Exponential(scale: float = 1.0, *, backend: Backend = "scalar") -> Any:
    """Construct an Exponential distribution for the selected backend.

    Parameters
    ----------
    scale : float, default=1.0
        Positive inverse-rate scale.
    backend : {"scalar", "numpy", "jax"}, default="scalar"
        Execution backend selected once during construction.

    Returns
    -------
    object
        Exponential distribution implemented by the selected backend.
    """

    return _module(backend).Exponential(scale=scale)


def Gamma(
    a: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    *,
    backend: Backend = "scalar",
) -> Any:
    """Construct a Gamma distribution for the selected backend.

    Parameters
    ----------
    a : float, default=1.0
        Positive shape parameter.
    loc : float, default=0.0
        Lower support boundary.
    scale : float, default=1.0
        Positive scale parameter.
    backend : {"scalar", "numpy", "jax"}, default="scalar"
        Execution backend selected once during construction.

    Returns
    -------
    object
        Gamma distribution implemented by the selected backend.
    """

    return _module(backend).Gamma(a=a, loc=loc, scale=scale)


def LogNormal(
    s: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    *,
    backend: Backend = "scalar",
) -> Any:
    """Construct a log-normal distribution for the selected backend.

    Parameters
    ----------
    s : float, default=1.0
        Positive shape parameter.
    loc : float, default=0.0
        Lower support boundary.
    scale : float, default=1.0
        Positive multiplicative scale.
    backend : {"scalar", "numpy", "jax"}, default="scalar"
        Execution backend selected once during construction.

    Returns
    -------
    object
        Log-normal distribution implemented by the selected backend.
    """

    return _module(backend).LogNormal(s=s, loc=loc, scale=scale)


def HalfNormal(
    loc: float = 0.0, scale: float = 1.0, *, backend: Backend = "scalar"
) -> Any:
    """Construct a half-normal distribution for the selected backend.

    Parameters
    ----------
    loc : float, default=0.0
        Lower support boundary.
    scale : float, default=1.0
        Positive scale parameter.
    backend : {"scalar", "numpy", "jax"}, default="scalar"
        Execution backend selected once during construction.

    Returns
    -------
    object
        Half-normal distribution implemented by the selected backend.
    """

    return _module(backend).HalfNormal(loc=loc, scale=scale)


def Cauchy(loc: float = 0.0, scale: float = 1.0, *, backend: Backend = "scalar") -> Any:
    """Construct a Cauchy distribution for the selected backend.

    Parameters
    ----------
    loc : float, default=0.0
        Distribution median and location.
    scale : float, default=1.0
        Positive scale parameter.
    backend : {"scalar", "numpy", "jax"}, default="scalar"
        Execution backend selected once during construction.

    Returns
    -------
    object
        Cauchy distribution implemented by the selected backend.
    """

    return _module(backend).Cauchy(loc=loc, scale=scale)


def Laplace(
    loc: float = 0.0, scale: float = 1.0, *, backend: Backend = "scalar"
) -> Any:
    """Construct a Laplace distribution for the selected backend.

    Parameters
    ----------
    loc : float, default=0.0
        Distribution mean and median.
    scale : float, default=1.0
        Positive scale parameter.
    backend : {"scalar", "numpy", "jax"}, default="scalar"
        Execution backend selected once during construction.

    Returns
    -------
    object
        Laplace distribution implemented by the selected backend.
    """

    return _module(backend).Laplace(loc=loc, scale=scale)


def Weibull(
    c: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    *,
    backend: Backend = "scalar",
) -> Any:
    """Construct a minimum Weibull distribution for the selected backend.

    Parameters
    ----------
    c : float, default=1.0
        Positive shape parameter.
    loc : float, default=0.0
        Lower support boundary.
    scale : float, default=1.0
        Positive scale parameter.
    backend : {"scalar", "numpy", "jax"}, default="scalar"
        Execution backend selected once during construction.

    Returns
    -------
    object
        Weibull distribution implemented by the selected backend.
    """

    return _module(backend).Weibull(c=c, loc=loc, scale=scale)


def TruncatedNormal(
    loc: float,
    scale: float,
    low: float,
    high: float,
    *,
    backend: Backend = "scalar",
) -> Any:
    """Construct a truncated Normal distribution for the selected backend.

    Parameters
    ----------
    loc : float
        Mean of the underlying Normal distribution.
    scale : float
        Positive standard deviation.
    low, high : float
        Ordered truncation boundaries.
    backend : {"scalar", "numpy", "jax"}, default="scalar"
        Execution backend selected once during construction.

    Returns
    -------
    object
        Truncated Normal distribution implemented by the selected backend.
    """

    return _module(backend).TruncatedNormal(loc=loc, scale=scale, low=low, high=high)


def TruncatedPowerLaw(
    alpha: float,
    low: float,
    high: float,
    *,
    backend: Backend = "scalar",
) -> Any:
    """Construct a truncated power-law distribution for the selected backend.

    Parameters
    ----------
    alpha : float
        Exponent in the density proportional to ``x**(-alpha)``.
    low, high : float
        Positive ordered support boundaries.
    backend : {"scalar", "numpy", "jax"}, default="scalar"
        Execution backend selected once during construction.

    Returns
    -------
    object
        Truncated power-law distribution implemented by the selected backend.
    """

    return _module(backend).TruncatedPowerLaw(alpha=alpha, low=low, high=high)


def TruncatedSine(*, backend: Backend = "scalar") -> Any:
    """Construct a truncated sine distribution for the selected backend.

    Parameters
    ----------
    backend : {"scalar", "numpy", "jax"}, default="scalar"
        Execution backend selected once during construction.

    Returns
    -------
    object
        Sine distribution on ``[0, pi / 2]`` for the selected backend.
    """

    return _module(backend).TruncatedSine()


def DiscreteUniform(low: int, high: int, *, backend: Backend = "scalar") -> Any:
    """Construct a discrete uniform distribution for the selected backend.

    Parameters
    ----------
    low, high : int
        Integer support boundaries defining ``[low, high)``.
    backend : {"scalar", "numpy", "jax"}, default="scalar"
        Execution backend selected once during construction.

    Returns
    -------
    object
        Discrete uniform distribution implemented by the selected backend.
    """

    return _module(backend).DiscreteUniform(low=low, high=high)
