"""JAX-traceable probability distributions.

Distribution parameters are validated once during construction. Evaluation
methods use :mod:`jax.numpy` and remain deliberately un-jitted so callers can
compile a complete prior transform or likelihood as one operation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import (
    betainc,
    gammainc,
    gammaincc,
    log_ndtr,
    ndtr,
    ndtri,
    xlog1py,
    xlogy,
)

__all__ = [
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
    "beta",
    "gamma",
    "normal",
    "randint",
    "truncsine",
    "uniform",
]

_LOG_TWO_PI = math.log(2.0 * math.pi)


class _TailMethods:
    """Provide traceable survival and logarithmic cumulative methods."""

    def sf(self, x: Any) -> jax.Array:
        """Evaluate the survival function.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        jax.Array
            Upper-tail probabilities.
        """

        survival = getattr(self, "_sf", None)
        if survival is not None:
            return survival(x)
        return jnp.clip(1.0 - self.cdf(x), 0.0, 1.0)

    def logcdf(self, x: Any) -> jax.Array:
        """Evaluate the logarithm of the cumulative distribution function.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        jax.Array
            Logarithmic lower-tail probabilities.
        """

        logarithm = getattr(self, "_logcdf", None)
        return logarithm(x) if logarithm is not None else jnp.log(self.cdf(x))

    def logsf(self, x: Any) -> jax.Array:
        """Evaluate the logarithm of the survival function.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        jax.Array
            Logarithmic upper-tail probabilities.
        """

        logarithm = getattr(self, "_logsf", None)
        return logarithm(x) if logarithm is not None else jnp.log(self.sf(x))


class _AnalyticDistribution(_TailMethods):
    """Expose the common JAX API for analytic distributions."""

    def logpdf(self, x: Any) -> jax.Array:
        """Evaluate the log-probability density.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        jax.Array
            Log-density at each point.
        """

        return self._log_density(x)

    def pdf(self, x: Any) -> jax.Array:
        """Evaluate the probability density.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        jax.Array
            Probability density at each point.
        """

        return jnp.exp(self._log_density(x))

    def cdf(self, x: Any) -> jax.Array:
        """Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        jax.Array
            Lower-tail probabilities.
        """

        return self._cumulative(x)

    def ppf(self, q: Any) -> jax.Array:
        """Evaluate the quantile function.

        Parameters
        ----------
        q : array-like
            Cumulative probabilities in ``[0, 1]``.

        Returns
        -------
        jax.Array
            Distribution quantiles.
        """

        q, valid = _quantile(q)
        return jnp.where(valid, self._inverse(q), jnp.nan)


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return value


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _quantile(q: Any) -> tuple[jax.Array, jax.Array]:
    values = jnp.asarray(q)
    return values, (values >= 0.0) & (values <= 1.0)


def _beta_ppf(q: Any, a: float, b: float) -> jax.Array:
    """Invert the regularized incomplete beta with traceable bisection."""

    q = jnp.asarray(q)
    low = jnp.zeros_like(q)
    high = jnp.ones_like(q)

    def step(_: int, bounds: tuple[jax.Array, jax.Array]):
        lower, upper = bounds
        middle = 0.5 * (lower + upper)
        move_lower = betainc(a, b, middle) < q
        return jnp.where(move_lower, middle, lower), jnp.where(
            move_lower, upper, middle
        )

    low, high = jax.lax.fori_loop(0, 64, step, (low, high))
    value = 0.5 * (low + high)
    value = jnp.where(q == 0.0, 0.0, value)
    return jnp.where(q == 1.0, 1.0, value)


def _gamma_ppf(q: Any, a: float) -> jax.Array:
    """Invert the regularized incomplete gamma with traceable bisection.

    Parameters
    ----------
    q : array-like
        Cumulative probabilities.
    a : float
        Positive shape parameter.

    Returns
    -------
    jax.Array
        Standard Gamma quantiles.
    """

    q = jnp.asarray(q)
    low = jnp.zeros_like(q)
    high = jnp.full_like(q, max(1.0, a))

    def expand(_: int, upper: jax.Array) -> jax.Array:
        """Expand upper brackets that remain below the target probability."""

        return jnp.where(gammainc(a, upper) < q, 2.0 * upper, upper)

    high = jax.lax.fori_loop(0, 64, expand, high)

    def bisect(
        _: int, bounds: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Apply one vectorized bisection iteration."""

        lower, upper = bounds
        middle = 0.5 * (lower + upper)
        move_lower = gammainc(a, middle) < q
        return jnp.where(move_lower, middle, lower), jnp.where(
            move_lower, upper, middle
        )

    low, high = jax.lax.fori_loop(0, 80, bisect, (low, high))
    value = 0.5 * (low + high)
    value = jnp.where(q == 0.0, 0.0, value)
    return jnp.where(q == 1.0, jnp.inf, value)


@dataclass(frozen=True, slots=True)
class Normal(_TailMethods):
    """Normal distribution with JAX-traceable methods."""

    loc: float = 0.0
    scale: float = 1.0
    _log_normalization: float = field(init=False, repr=False)
    _inverse_scale: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        loc = _finite(self.loc, "loc")
        scale = _positive(self.scale, "scale")
        object.__setattr__(self, "loc", loc)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "_inverse_scale", 1.0 / scale)
        object.__setattr__(
            self, "_log_normalization", -0.5 * _LOG_TWO_PI - math.log(scale)
        )

    @property
    def mean(self) -> float:
        return self.loc

    @property
    def median(self) -> float:
        return self.loc

    def logpdf(self, x: Any, norm: bool = True) -> jax.Array:
        z = (jnp.asarray(x) - self.loc) * self._inverse_scale
        value = -0.5 * z * z
        return value + self._log_normalization if norm else value

    def pdf(self, x: Any, norm: bool = True) -> jax.Array:
        value = jnp.exp(-0.5 * ((jnp.asarray(x) - self.loc) * self._inverse_scale) ** 2)
        return value * math.exp(self._log_normalization) if norm else value

    def cdf(self, x: Any) -> jax.Array:
        return ndtr((jnp.asarray(x) - self.loc) * self._inverse_scale)

    def _sf(self, x: Any) -> jax.Array:
        return ndtr(-(jnp.asarray(x) - self.loc) * self._inverse_scale)

    def _logcdf(self, x: Any) -> jax.Array:
        return log_ndtr((jnp.asarray(x) - self.loc) * self._inverse_scale)

    def _logsf(self, x: Any) -> jax.Array:
        return log_ndtr(-(jnp.asarray(x) - self.loc) * self._inverse_scale)

    def ppf(self, q: Any) -> jax.Array:
        q, valid = _quantile(q)
        value = self.loc + self.scale * ndtri(q)
        return jnp.where(valid, value, jnp.nan)


@dataclass(frozen=True, slots=True)
class Uniform(_TailMethods):
    """Continuous uniform distribution on ``[loc, loc + scale]``."""

    loc: float = 0.0
    scale: float = 1.0
    high: float = field(init=False)
    _density: float = field(init=False, repr=False)
    _log_density: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        loc = _finite(self.loc, "loc")
        scale = _positive(self.scale, "scale")
        object.__setattr__(self, "loc", loc)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "high", loc + scale)
        object.__setattr__(self, "_density", 1.0 / scale)
        object.__setattr__(self, "_log_density", -math.log(scale))

    @property
    def mean(self) -> float:
        return self.loc + 0.5 * self.scale

    @property
    def median(self) -> float:
        return self.mean

    def pdf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.where((x >= self.loc) & (x <= self.high), self._density, 0.0)

    def logpdf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.where(
            (x >= self.loc) & (x <= self.high), self._log_density, -jnp.inf
        )

    def cdf(self, x: Any) -> jax.Array:
        return jnp.clip((jnp.asarray(x) - self.loc) / self.scale, 0.0, 1.0)

    def _sf(self, x: Any) -> jax.Array:
        return jnp.clip((self.high - jnp.asarray(x)) / self.scale, 0.0, 1.0)

    def ppf(self, q: Any) -> jax.Array:
        q, valid = _quantile(q)
        return jnp.where(valid, self.loc + q * self.scale, jnp.nan)


@dataclass(frozen=True, slots=True)
class Beta(_TailMethods):
    """Beta distribution transformed to ``[loc, loc + scale]``."""

    a: float = 1.0
    b: float = 1.0
    loc: float = 0.0
    scale: float = 1.0
    high: float = field(init=False)
    _log_normalization: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        a = _positive(self.a, "a")
        b = _positive(self.b, "b")
        loc = _finite(self.loc, "loc")
        scale = _positive(self.scale, "scale")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "loc", loc)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "high", loc + scale)
        object.__setattr__(
            self,
            "_log_normalization",
            -math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b) - math.log(scale),
        )

    @property
    def mean(self) -> float:
        return self.loc + self.scale * self.a / (self.a + self.b)

    @property
    def median(self) -> jax.Array:
        return self.ppf(0.5)

    def logpdf(self, x: Any, norm: bool = True) -> jax.Array:
        y = (jnp.asarray(x) - self.loc) / self.scale
        value = xlogy(self.a - 1.0, y) + xlog1py(self.b - 1.0, -y)
        if norm:
            value = value + self._log_normalization
        return jnp.where((y >= 0.0) & (y <= 1.0), value, -jnp.inf)

    def pdf(self, x: Any, norm: bool = True) -> jax.Array:
        return jnp.exp(self.logpdf(x, norm=norm))

    def cdf(self, x: Any) -> jax.Array:
        y = (jnp.asarray(x) - self.loc) / self.scale
        value = betainc(self.a, self.b, jnp.clip(y, 0.0, 1.0))
        return jnp.where(y <= 0.0, 0.0, jnp.where(y >= 1.0, 1.0, value))

    def _sf(self, x: Any) -> jax.Array:
        y = (jnp.asarray(x) - self.loc) / self.scale
        value = betainc(self.b, self.a, jnp.clip(1.0 - y, 0.0, 1.0))
        return jnp.where(y <= 0.0, 1.0, jnp.where(y >= 1.0, 0.0, value))

    def ppf(self, q: Any) -> jax.Array:
        q, valid = _quantile(q)
        value = self.loc + self.scale * _beta_ppf(q, self.a, self.b)
        return jnp.where(valid, value, jnp.nan)


@dataclass(frozen=True, slots=True)
class Exponential(_TailMethods):
    """Exponential distribution using the SciPy ``scale`` convention."""

    scale: float = 1.0
    _inverse_scale: float = field(init=False, repr=False)
    _log_scale: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        scale = _positive(self.scale, "scale")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "_inverse_scale", 1.0 / scale)
        object.__setattr__(self, "_log_scale", math.log(scale))

    @property
    def mean(self) -> float:
        return self.scale

    @property
    def median(self) -> float:
        return self.scale * math.log(2.0)

    def logpdf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.where(x >= 0.0, -x * self._inverse_scale - self._log_scale, -jnp.inf)

    def pdf(self, x: Any) -> jax.Array:
        return jnp.exp(self.logpdf(x))

    def cdf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.where(x > 0.0, -jnp.expm1(-x * self._inverse_scale), 0.0)

    def _sf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.where(x >= 0.0, jnp.exp(-x * self._inverse_scale), 1.0)

    def _logsf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.where(x >= 0.0, -x * self._inverse_scale, 0.0)

    def ppf(self, q: Any) -> jax.Array:
        q, valid = _quantile(q)
        return jnp.where(valid, -self.scale * jnp.log1p(-q), jnp.nan)


@dataclass(frozen=True, slots=True)
class TruncatedNormal(_TailMethods):
    """Normal distribution restricted to ``[low, high]``."""

    loc: float
    scale: float
    low: float
    high: float
    _cdf_low: float = field(init=False, repr=False)
    _cdf_width: float = field(init=False, repr=False)
    _log_normalization: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        loc = _finite(self.loc, "loc")
        scale = _positive(self.scale, "scale")
        low = float(self.low)
        high = float(self.high)
        if math.isnan(low) or math.isnan(high) or low >= high:
            raise ValueError("low must be less than high")

        def normal_cdf(value: float) -> float:
            return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

        cdf_low = normal_cdf((low - loc) / scale)
        cdf_high = normal_cdf((high - loc) / scale)
        width = cdf_high - cdf_low
        if width <= 0.0:
            message = "truncation interval has no representable probability mass"
            raise ValueError(message)
        object.__setattr__(self, "loc", loc)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "_cdf_low", cdf_low)
        object.__setattr__(self, "_cdf_width", width)
        object.__setattr__(
            self,
            "_log_normalization",
            -0.5 * _LOG_TWO_PI - math.log(scale) - math.log(width),
        )

    def logpdf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        z = (x - self.loc) / self.scale
        return jnp.where(
            (x >= self.low) & (x <= self.high),
            -0.5 * z * z + self._log_normalization,
            -jnp.inf,
        )

    def pdf(self, x: Any) -> jax.Array:
        return jnp.exp(self.logpdf(x))

    def cdf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        value = ndtr((x - self.loc) / self.scale) - self._cdf_low
        return jnp.clip(value / self._cdf_width, 0.0, 1.0)

    def ppf(self, q: Any) -> jax.Array:
        q, valid = _quantile(q)
        probability = self._cdf_low + q * self._cdf_width
        value = self.loc + self.scale * ndtri(probability)
        value = jnp.where(q == 0.0, self.low, value)
        value = jnp.where(q == 1.0, self.high, value)
        return jnp.where(valid, value, jnp.nan)


@dataclass(frozen=True, slots=True)
class Gamma(_TailMethods):
    """Gamma distribution with JAX-traceable methods.

    Parameters
    ----------
    a : float, default=1.0
        Positive shape parameter.
    loc : float, default=0.0
        Lower support boundary.
    scale : float, default=1.0
        Positive scale parameter.
    """

    a: float = 1.0
    loc: float = 0.0
    scale: float = 1.0
    _log_normalization: float = field(init=False, repr=False)
    _inverse_scale: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate parameters and cache normalization terms."""

        a = _positive(self.a, "a")
        loc = _finite(self.loc, "loc")
        scale = _positive(self.scale, "scale")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "loc", loc)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "_inverse_scale", 1.0 / scale)
        object.__setattr__(
            self, "_log_normalization", -math.lgamma(a) - math.log(scale)
        )

    @property
    def mean(self) -> float:
        """Return the distribution mean."""

        return self.loc + self.a * self.scale

    @property
    def median(self) -> jax.Array:
        """Return the distribution median."""

        return self.ppf(0.5)

    def logpdf(self, x: Any, norm: bool = True) -> jax.Array:
        """Evaluate the log-probability density.

        Parameters
        ----------
        x : array-like
            Evaluation points.
        norm : bool, default=True
            Include the normalization constant when true.

        Returns
        -------
        jax.Array
            Log-density at each evaluation point.
        """

        y = (jnp.asarray(x) - self.loc) * self._inverse_scale
        value = xlogy(self.a - 1.0, y) - y
        if norm:
            value = value + self._log_normalization
        return jnp.where(y >= 0.0, value, -jnp.inf)

    def pdf(self, x: Any, norm: bool = True) -> jax.Array:
        """Evaluate the probability density.

        Parameters
        ----------
        x : array-like
            Evaluation points.
        norm : bool, default=True
            Include the normalization constant when true.

        Returns
        -------
        jax.Array
            Probability density at each evaluation point.
        """

        return jnp.exp(self.logpdf(x, norm=norm))

    def cdf(self, x: Any) -> jax.Array:
        """Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        jax.Array
            Cumulative probability at each evaluation point.
        """

        y = (jnp.asarray(x) - self.loc) * self._inverse_scale
        return jnp.where(y > 0.0, gammainc(self.a, y), 0.0)

    def _sf(self, x: Any) -> jax.Array:
        y = (jnp.asarray(x) - self.loc) * self._inverse_scale
        return jnp.where(y > 0.0, gammaincc(self.a, y), 1.0)

    def ppf(self, q: Any) -> jax.Array:
        """Evaluate the quantile function.

        Parameters
        ----------
        q : array-like
            Cumulative probabilities in ``[0, 1]``.

        Returns
        -------
        jax.Array
            Distribution quantiles.
        """

        q, valid = _quantile(q)
        value = self.loc + self.scale * _gamma_ppf(q, self.a)
        return jnp.where(valid, value, jnp.nan)


@dataclass(frozen=True, slots=True)
class TruncatedPowerLaw(_TailMethods):
    """Density proportional to ``x**(-alpha)`` on ``[low, high]``."""

    alpha: float
    low: float
    high: float
    _power: float = field(init=False, repr=False)
    _log_uniform: bool = field(init=False, repr=False)
    _lower_term: float = field(init=False, repr=False)
    _width_term: float = field(init=False, repr=False)
    _log_normalization: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        alpha = _finite(self.alpha, "alpha")
        low = _positive(self.low, "low")
        high = _positive(self.high, "high")
        if low >= high:
            raise ValueError("low must be less than high")
        power = 1.0 - alpha
        log_uniform = abs(power) < 1e-12
        if log_uniform:
            lower_term = math.log(low)
            width_term = math.log(high) - lower_term
            log_normalization = -math.log(width_term)
        else:
            lower_term = low**power
            width_term = high**power - lower_term
            log_normalization = math.log(abs(power / width_term))
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "_power", power)
        object.__setattr__(self, "_log_uniform", log_uniform)
        object.__setattr__(self, "_lower_term", lower_term)
        object.__setattr__(self, "_width_term", width_term)
        object.__setattr__(self, "_log_normalization", log_normalization)

    def logpdf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        value = self._log_normalization - self.alpha * jnp.log(x)
        return jnp.where((x >= self.low) & (x <= self.high), value, -jnp.inf)

    def pdf(self, x: Any) -> jax.Array:
        return jnp.exp(self.logpdf(x))

    def cdf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        if self._log_uniform:
            value = (jnp.log(x) - self._lower_term) / self._width_term
        else:
            value = (x**self._power - self._lower_term) / self._width_term
        return jnp.where(x <= self.low, 0.0, jnp.where(x >= self.high, 1.0, value))

    def ppf(self, q: Any) -> jax.Array:
        q, valid = _quantile(q)
        if self._log_uniform:
            value = jnp.exp(self._lower_term + q * self._width_term)
        else:
            value = (self._lower_term + q * self._width_term) ** (1.0 / self._power)
        return jnp.where(valid, value, jnp.nan)


@dataclass(frozen=True, slots=True)
class TruncatedSine(_TailMethods):
    """Sine density on the interval ``[0, pi / 2]``."""

    @property
    def mean(self) -> float:
        return 1.0

    @property
    def median(self) -> float:
        return math.pi / 3.0

    def pdf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.where((x >= 0.0) & (x <= math.pi / 2.0), jnp.sin(x), 0.0)

    def logpdf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        value = jnp.log(jnp.sin(x))
        return jnp.where((x > 0.0) & (x <= math.pi / 2.0), value, -jnp.inf)

    def cdf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.where(
            x <= 0.0,
            0.0,
            jnp.where(x >= math.pi / 2.0, 1.0, 1.0 - jnp.cos(x)),
        )

    def ppf(self, q: Any) -> jax.Array:
        q, valid = _quantile(q)
        return jnp.where(valid, jnp.arccos(1.0 - q), jnp.nan)


@dataclass(frozen=True, slots=True)
class LogNormal(_AnalyticDistribution):
    """Log-normal distribution using SciPy's shape, location, and scale."""

    s: float = 1.0
    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "s", _positive(self.s, "s"))
        object.__setattr__(self, "loc", _finite(self.loc, "loc"))
        object.__setattr__(self, "scale", _positive(self.scale, "scale"))

    @property
    def mean(self) -> float:
        return self.loc + self.scale * math.exp(0.5 * self.s**2)

    @property
    def median(self) -> float:
        return self.loc + self.scale

    def _z(self, x: Any) -> jax.Array:
        return jnp.log((jnp.asarray(x) - self.loc) / self.scale) / self.s

    def _log_density(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        z = self._z(x)
        value = -0.5 * z * z - jnp.log(x - self.loc)
        value -= math.log(self.s) + 0.5 * _LOG_TWO_PI
        return jnp.where(x > self.loc, value, -jnp.inf)

    def _cumulative(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.where(x > self.loc, ndtr(self._z(x)), 0.0)

    def _sf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.where(x > self.loc, ndtr(-self._z(x)), 1.0)

    def _logcdf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.where(x > self.loc, log_ndtr(self._z(x)), -jnp.inf)

    def _logsf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.where(x > self.loc, log_ndtr(-self._z(x)), 0.0)

    def _inverse(self, q: jax.Array) -> jax.Array:
        return self.loc + self.scale * jnp.exp(self.s * ndtri(q))


@dataclass(frozen=True, slots=True)
class HalfNormal(_AnalyticDistribution):
    """Half-normal distribution using SciPy's location and scale convention."""

    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "loc", _finite(self.loc, "loc"))
        object.__setattr__(self, "scale", _positive(self.scale, "scale"))

    @property
    def mean(self) -> float:
        return self.loc + self.scale * math.sqrt(2.0 / math.pi)

    @property
    def median(self) -> jax.Array:
        return self.ppf(0.5)

    def _log_density(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        z = (x - self.loc) / self.scale
        value = -0.5 * z * z + 0.5 * math.log(2.0 / math.pi)
        return jnp.where(x >= self.loc, value - math.log(self.scale), -jnp.inf)

    def _cumulative(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        z = (x - self.loc) / self.scale
        return jnp.where(x > self.loc, 2.0 * ndtr(z) - 1.0, 0.0)

    def _sf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        z = (x - self.loc) / self.scale
        return jnp.where(x > self.loc, 2.0 * ndtr(-z), 1.0)

    def _logsf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        z = (x - self.loc) / self.scale
        return jnp.where(x > self.loc, math.log(2.0) + log_ndtr(-z), 0.0)

    def _inverse(self, q: jax.Array) -> jax.Array:
        return self.loc + self.scale * ndtri(0.5 * (q + 1.0))


@dataclass(frozen=True, slots=True)
class Cauchy(_AnalyticDistribution):
    """Cauchy distribution using SciPy's location and scale convention."""

    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "loc", _finite(self.loc, "loc"))
        object.__setattr__(self, "scale", _positive(self.scale, "scale"))

    @property
    def mean(self) -> float:
        return math.nan

    @property
    def median(self) -> float:
        return self.loc

    def _log_density(self, x: Any) -> jax.Array:
        z = (jnp.asarray(x) - self.loc) / self.scale
        return -math.log(math.pi * self.scale) - jnp.log1p(z * z)

    def _cumulative(self, x: Any) -> jax.Array:
        z = (jnp.asarray(x) - self.loc) / self.scale
        return 0.5 + jnp.arctan(z) / math.pi

    def _sf(self, x: Any) -> jax.Array:
        z = (jnp.asarray(x) - self.loc) / self.scale
        return jnp.where(
            z > 0.0,
            jnp.arctan(1.0 / z) / math.pi,
            0.5 - jnp.arctan(z) / math.pi,
        )

    def _inverse(self, q: jax.Array) -> jax.Array:
        value = self.loc + self.scale * jnp.tan(math.pi * (q - 0.5))
        value = jnp.where(q == 0.0, -jnp.inf, value)
        return jnp.where(q == 1.0, jnp.inf, value)


@dataclass(frozen=True, slots=True)
class Laplace(_AnalyticDistribution):
    """Laplace distribution using SciPy's location and scale convention."""

    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "loc", _finite(self.loc, "loc"))
        object.__setattr__(self, "scale", _positive(self.scale, "scale"))

    @property
    def mean(self) -> float:
        return self.loc

    @property
    def median(self) -> float:
        return self.loc

    def _log_density(self, x: Any) -> jax.Array:
        return -jnp.abs(jnp.asarray(x) - self.loc) / self.scale - math.log(
            2.0 * self.scale
        )

    def _cumulative(self, x: Any) -> jax.Array:
        z = (jnp.asarray(x) - self.loc) / self.scale
        return jnp.where(z <= 0.0, 0.5 * jnp.exp(z), 1.0 - 0.5 * jnp.exp(-z))

    def _sf(self, x: Any) -> jax.Array:
        z = (jnp.asarray(x) - self.loc) / self.scale
        return jnp.where(z <= 0.0, 1.0 - 0.5 * jnp.exp(z), 0.5 * jnp.exp(-z))

    def _logcdf(self, x: Any) -> jax.Array:
        z = (jnp.asarray(x) - self.loc) / self.scale
        return jnp.where(z <= 0.0, math.log(0.5) + z, jnp.log1p(-0.5 * jnp.exp(-z)))

    def _logsf(self, x: Any) -> jax.Array:
        z = (jnp.asarray(x) - self.loc) / self.scale
        return jnp.where(z <= 0.0, jnp.log1p(-0.5 * jnp.exp(z)), math.log(0.5) - z)

    def _inverse(self, q: jax.Array) -> jax.Array:
        return jnp.where(
            q < 0.5,
            self.loc + self.scale * jnp.log(2.0 * q),
            self.loc - self.scale * jnp.log(2.0 * (1.0 - q)),
        )


@dataclass(frozen=True, slots=True)
class Weibull(_AnalyticDistribution):
    """Minimum Weibull distribution using SciPy's shape, location, and scale."""

    c: float = 1.0
    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "c", _positive(self.c, "c"))
        object.__setattr__(self, "loc", _finite(self.loc, "loc"))
        object.__setattr__(self, "scale", _positive(self.scale, "scale"))

    @property
    def mean(self) -> float:
        return self.loc + self.scale * math.gamma(1.0 + 1.0 / self.c)

    @property
    def median(self) -> float:
        return self.loc + self.scale * math.log(2.0) ** (1.0 / self.c)

    def _log_density(self, x: Any) -> jax.Array:
        z = (jnp.asarray(x) - self.loc) / self.scale
        value = math.log(self.c / self.scale)
        value += (self.c - 1.0) * jnp.log(z) - z**self.c
        boundary = (
            -jnp.inf
            if self.c > 1.0
            else (jnp.inf if self.c < 1.0 else -math.log(self.scale))
        )
        return jnp.where(z < 0.0, -jnp.inf, jnp.where(z == 0.0, boundary, value))

    def _cumulative(self, x: Any) -> jax.Array:
        z = (jnp.asarray(x) - self.loc) / self.scale
        return jnp.where(z > 0.0, -jnp.expm1(-(z**self.c)), 0.0)

    def _sf(self, x: Any) -> jax.Array:
        z = (jnp.asarray(x) - self.loc) / self.scale
        return jnp.where(z > 0.0, jnp.exp(-(z**self.c)), 1.0)

    def _logsf(self, x: Any) -> jax.Array:
        z = (jnp.asarray(x) - self.loc) / self.scale
        return jnp.where(z > 0.0, -(z**self.c), 0.0)

    def _inverse(self, q: jax.Array) -> jax.Array:
        return self.loc + self.scale * (-jnp.log1p(-q)) ** (1.0 / self.c)


@dataclass(frozen=True, slots=True)
class DiscreteUniform(_TailMethods):
    """Discrete uniform distribution on integers in ``[low, high)``."""

    low: int
    high: int
    _count: int = field(init=False, repr=False)
    _mass: float = field(init=False, repr=False)
    _log_mass: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.low, bool) or not isinstance(self.low, int):
            raise TypeError("low must be an integer")
        if isinstance(self.high, bool) or not isinstance(self.high, int):
            raise TypeError("high must be an integer")
        if self.low >= self.high:
            raise ValueError("low must be less than high")
        count = self.high - self.low
        object.__setattr__(self, "_count", count)
        object.__setattr__(self, "_mass", 1.0 / count)
        object.__setattr__(self, "_log_mass", -math.log(count))

    @property
    def mean(self) -> float:
        return 0.5 * (self.low + self.high - 1)

    @property
    def median(self) -> jax.Array:
        return self.ppf(0.5)

    def pdf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        valid = (
            jnp.isfinite(x) & (x >= self.low) & (x < self.high) & (x == jnp.floor(x))
        )
        return jnp.where(valid, self._mass, 0.0)

    pmf = pdf

    def logpdf(self, x: Any) -> jax.Array:
        return jnp.where(self.pdf(x) > 0.0, self._log_mass, -jnp.inf)

    logpmf = logpdf

    def cdf(self, x: Any) -> jax.Array:
        x = jnp.asarray(x)
        value = (jnp.floor(x) - self.low + 1) / self._count
        return jnp.clip(value, 0.0, 1.0)

    def ppf(self, q: Any) -> jax.Array:
        q, valid = _quantile(q)
        value = self.low + jnp.ceil(q * self._count) - 1
        value = jnp.where(q == 0.0, self.low, value)
        return jnp.where(valid, value, jnp.nan)


normal = Normal
uniform = Uniform
beta = Beta
gamma = Gamma
truncsine = TruncatedSine
randint = DiscreteUniform
