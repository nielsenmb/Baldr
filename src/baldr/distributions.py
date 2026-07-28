"""Backend-selecting public distribution constructors."""

from __future__ import annotations

from typing import Any, Literal

from baldr import scalar

Backend = Literal["scalar", "numpy", "jax"]

__all__ = [
    "Backend",
    "Beta",
    "DiscreteUniform",
    "Exponential",
    "Gamma",
    "Normal",
    "TruncatedNormal",
    "TruncatedPowerLaw",
    "TruncatedSine",
    "Uniform",
]


def _module(backend: Backend) -> Any:
    if backend == "scalar":
        return scalar
    if backend == "numpy":
        from baldr import numpy

        return numpy
    if backend == "jax":
        from baldr import jax

        return jax
    raise ValueError("backend must be 'scalar', 'numpy', or 'jax'")


def Normal(
    loc: float = 0.0, scale: float = 1.0, *, backend: Backend = "scalar"
) -> Any:
    """Construct a Normal distribution for the selected backend."""

    return _module(backend).Normal(loc=loc, scale=scale)


def Uniform(
    loc: float = 0.0, scale: float = 1.0, *, backend: Backend = "scalar"
) -> Any:
    """Construct a continuous uniform distribution for the selected backend."""

    return _module(backend).Uniform(loc=loc, scale=scale)


def Beta(
    a: float = 1.0,
    b: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    *,
    backend: Backend = "scalar",
) -> Any:
    """Construct a transformed Beta distribution for the selected backend."""

    return _module(backend).Beta(a=a, b=b, loc=loc, scale=scale)


def Exponential(scale: float = 1.0, *, backend: Backend = "scalar") -> Any:
    """Construct an Exponential distribution for the selected backend."""

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


def TruncatedNormal(
    loc: float,
    scale: float,
    low: float,
    high: float,
    *,
    backend: Backend = "scalar",
) -> Any:
    """Construct a truncated Normal distribution for the selected backend."""

    return _module(backend).TruncatedNormal(
        loc=loc, scale=scale, low=low, high=high
    )


def TruncatedPowerLaw(
    alpha: float,
    low: float,
    high: float,
    *,
    backend: Backend = "scalar",
) -> Any:
    """Construct a truncated power-law distribution for the selected backend."""

    return _module(backend).TruncatedPowerLaw(alpha=alpha, low=low, high=high)


def TruncatedSine(*, backend: Backend = "scalar") -> Any:
    """Construct a truncated sine distribution for the selected backend."""

    return _module(backend).TruncatedSine()


def DiscreteUniform(
    low: int, high: int, *, backend: Backend = "scalar"
) -> Any:
    """Construct a discrete uniform distribution for the selected backend."""

    return _module(backend).DiscreteUniform(low=low, high=high)
