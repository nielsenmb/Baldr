"""Backend-neutral wrappers for user-supplied distribution functions."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = ["CallableDistribution", "distribution"]


class _DistributionFunction(Protocol):
    """Protocol for scalar or array distribution functions."""

    def __call__(self, value: Any) -> Any:
        """Evaluate the function.

        Parameters
        ----------
        value : object
            Scalar or array-like input accepted by the wrapped function.

        Returns
        -------
        object
            Function result.
        """

        ...


@dataclass(frozen=True, slots=True)
class CallableDistribution:
    """Wrap callables with Baldr's distribution interface.

    The callables are stored unchanged, so their scalar, NumPy, or JAX
    behaviour is preserved. Unlike PBjam's legacy wrapper, construction does
    not estimate the mean through numerical integration.

    Parameters
    ----------
    ppf, pdf, logpdf, cdf : callable
        Quantile, density, log-density, and cumulative functions.
    rng : object, optional
        Random-number generator exposing ``uniform(0, 1)`` or ``random()``.
        The standard-library generator is used when omitted.
    mean : object, optional
        Known distribution mean. No estimate is made when omitted.
    median : object, optional
        Known distribution median. Defaults to ``ppf(0.5)``.
    """

    ppf: _DistributionFunction
    pdf: _DistributionFunction
    logpdf: _DistributionFunction
    cdf: _DistributionFunction
    rng: Any = None
    mean: Any = None
    median: Any = None

    def __post_init__(self) -> None:
        """Validate callables and derive the median when it is not supplied."""

        for name in ("ppf", "pdf", "logpdf", "cdf"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        if self.median is None:
            object.__setattr__(self, "median", self.ppf(0.5))

    def rv(self) -> Any:
        """Draw one random variate through the wrapped quantile function.

        Returns
        -------
        object
            Random variate with the scalar or array type returned by ``ppf``.
        """

        if self.rng is None:
            probability = random.random()
        elif hasattr(self.rng, "uniform"):
            probability = self.rng.uniform(0.0, 1.0)
        elif hasattr(self.rng, "random"):
            probability = self.rng.random()
        else:
            raise TypeError("rng must provide uniform(0, 1) or random()")
        return self.ppf(probability)


# Transitional alias for PBjam's lower-case class name.
distribution = CallableDistribution
