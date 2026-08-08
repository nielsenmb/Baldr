"""Broadcasting NumPy probability distributions.

Parameters are validated once when a frozen distribution object is constructed.
Evaluation methods accept scalars, sequences, and broadcastable NumPy arrays.
SciPy supplies special functions that NumPy does not implement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import (
    betainc,
    betaincinv,
    betaln,
    gammainc,
    gammaincc,
    gammaincinv,
    gammaln,
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
    "StudentT",
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
    """Provide survival and logarithmic cumulative methods."""

    def sf(self, x: Any) -> np.ndarray:
        """Evaluate the survival function.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        numpy.ndarray
            Upper-tail probabilities.
        """

        survival = getattr(self, "_sf", None)
        if survival is not None:
            return survival(x)
        return np.clip(1.0 - self.cdf(x), 0.0, 1.0)

    def logcdf(self, x: Any) -> np.ndarray:
        """Evaluate the logarithm of the cumulative distribution function.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        numpy.ndarray
            Logarithmic lower-tail probabilities.
        """

        logarithm = getattr(self, "_logcdf", None)
        if logarithm is not None:
            return logarithm(x)
        with np.errstate(divide="ignore"):
            return np.log(self.cdf(x))

    def logsf(self, x: Any) -> np.ndarray:
        """Evaluate the logarithm of the survival function.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        numpy.ndarray
            Logarithmic upper-tail probabilities.
        """

        logarithm = getattr(self, "_logsf", None)
        if logarithm is not None:
            return logarithm(x)
        with np.errstate(divide="ignore"):
            return np.log(self.sf(x))


class _AnalyticDistribution(_TailMethods):
    """Expose the common array API for analytic distributions."""

    def logpdf(self, x: Any) -> np.ndarray:
        """Evaluate the log-probability density.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        numpy.ndarray
            Log-density at each point.
        """

        return self._log_density(x)

    def pdf(self, x: Any) -> np.ndarray:
        """Evaluate the probability density.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        numpy.ndarray
            Probability density at each point.
        """

        return np.exp(self._log_density(x))

    def cdf(self, x: Any) -> np.ndarray:
        """Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        numpy.ndarray
            Lower-tail probabilities.
        """

        return self._cumulative(x)

    def ppf(self, q: Any) -> np.ndarray:
        """Evaluate the quantile function.

        Parameters
        ----------
        q : array-like
            Cumulative probabilities in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray
            Distribution quantiles.
        """

        q, valid = _valid_quantile(q)
        return np.where(valid, self._quantile(q), np.nan)


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


def _valid_quantile(q: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(q)
    return values, (values >= 0.0) & (values <= 1.0)


@dataclass(frozen=True, slots=True)
class Normal(_TailMethods):
    """Normal distribution with broadcasting evaluation methods."""

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

    def logpdf(self, x: Any, norm: bool = True) -> np.ndarray:
        z = (np.asarray(x) - self.loc) * self._inverse_scale
        value = -0.5 * z * z
        return value + self._log_normalization if norm else value

    def pdf(self, x: Any, norm: bool = True) -> np.ndarray:
        value = np.exp(-0.5 * ((np.asarray(x) - self.loc) * self._inverse_scale) ** 2)
        return value * math.exp(self._log_normalization) if norm else value

    def cdf(self, x: Any) -> np.ndarray:
        return ndtr((np.asarray(x) - self.loc) * self._inverse_scale)

    def _sf(self, x: Any) -> np.ndarray:
        return ndtr(-(np.asarray(x) - self.loc) * self._inverse_scale)

    def _logcdf(self, x: Any) -> np.ndarray:
        return log_ndtr((np.asarray(x) - self.loc) * self._inverse_scale)

    def _logsf(self, x: Any) -> np.ndarray:
        return log_ndtr(-(np.asarray(x) - self.loc) * self._inverse_scale)

    def ppf(self, q: Any) -> np.ndarray:
        q, valid = _valid_quantile(q)
        value = self.loc + self.scale * ndtri(q)
        return np.where(valid, value, np.nan)


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

    def pdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        return np.where((x >= self.loc) & (x <= self.high), self._density, 0.0)

    def logpdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        return np.where((x >= self.loc) & (x <= self.high), self._log_density, -np.inf)

    def cdf(self, x: Any) -> np.ndarray:
        return np.clip((np.asarray(x) - self.loc) / self.scale, 0.0, 1.0)

    def _sf(self, x: Any) -> np.ndarray:
        return np.clip((self.high - np.asarray(x)) / self.scale, 0.0, 1.0)

    def ppf(self, q: Any) -> np.ndarray:
        q, valid = _valid_quantile(q)
        return np.where(valid, self.loc + q * self.scale, np.nan)


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
            self, "_log_normalization", -float(betaln(a, b)) - math.log(scale)
        )

    @property
    def mean(self) -> float:
        return self.loc + self.scale * self.a / (self.a + self.b)

    @property
    def median(self) -> float:
        return float(self.ppf(0.5))

    def logpdf(self, x: Any, norm: bool = True) -> np.ndarray:
        y = (np.asarray(x) - self.loc) / self.scale
        value = xlogy(self.a - 1.0, y) + xlog1py(self.b - 1.0, -y)
        if norm:
            value = value + self._log_normalization
        return np.where((y >= 0.0) & (y <= 1.0), value, -np.inf)

    def pdf(self, x: Any, norm: bool = True) -> np.ndarray:
        return np.exp(self.logpdf(x, norm=norm))

    def cdf(self, x: Any) -> np.ndarray:
        y = (np.asarray(x) - self.loc) / self.scale
        return np.where(
            y <= 0.0, 0.0, np.where(y >= 1.0, 1.0, betainc(self.a, self.b, y))
        )

    def _sf(self, x: Any) -> np.ndarray:
        y = (np.asarray(x) - self.loc) / self.scale
        return np.where(
            y <= 0.0,
            1.0,
            np.where(y >= 1.0, 0.0, betainc(self.b, self.a, 1.0 - y)),
        )

    def ppf(self, q: Any) -> np.ndarray:
        q, valid = _valid_quantile(q)
        value = self.loc + self.scale * betaincinv(self.a, self.b, q)
        return np.where(valid, value, np.nan)


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

    def logpdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        return np.where(x >= 0.0, -x * self._inverse_scale - self._log_scale, -np.inf)

    def pdf(self, x: Any) -> np.ndarray:
        return np.exp(self.logpdf(x))

    def cdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        return np.where(x > 0.0, -np.expm1(-x * self._inverse_scale), 0.0)

    def _sf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        return np.where(x >= 0.0, np.exp(-x * self._inverse_scale), 1.0)

    def _logsf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        return np.where(x >= 0.0, -x * self._inverse_scale, 0.0)

    def ppf(self, q: Any) -> np.ndarray:
        q, valid = _valid_quantile(q)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = -self.scale * np.log1p(-q)
        return np.where(valid, value, np.nan)


@dataclass(frozen=True, slots=True)
class Gamma(_TailMethods):
    """Gamma distribution with broadcasting NumPy methods.

    Parameters
    ----------
    a : array-like, default=1.0
        Positive shape parameter. Array parameters broadcast with ``scale`` and
        with evaluation points.
    loc : float, default=0.0
        Lower support boundary.
    scale : array-like, default=1.0
        Positive scale parameter. Array parameters broadcast with ``a`` and
        with evaluation points.
    """

    a: Any = 1.0
    loc: float = 0.0
    scale: Any = 1.0
    _log_normalization: np.ndarray = field(init=False, repr=False)
    _inverse_scale: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate parameters and cache normalization terms."""

        a = np.asarray(self.a, dtype=float)
        loc = _finite(self.loc, "loc")
        scale = np.asarray(self.scale, dtype=float)
        if np.any(~np.isfinite(a)) or np.any(a <= 0.0):
            raise ValueError("a must contain only finite positive values")
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("scale must contain only finite positive values")
        try:
            a, scale = np.broadcast_arrays(a, scale)
        except ValueError as error:
            raise ValueError("a and scale must broadcast together") from error
        scalar_parameters = a.ndim == 0 and scale.ndim == 0
        object.__setattr__(self, "a", float(a) if scalar_parameters else a)
        object.__setattr__(self, "loc", loc)
        object.__setattr__(self, "scale", float(scale) if scalar_parameters else scale)
        object.__setattr__(self, "_inverse_scale", 1.0 / self.scale)
        object.__setattr__(
            self, "_log_normalization", -gammaln(self.a) - np.log(self.scale)
        )

    @property
    def mean(self) -> Any:
        """Return the distribution mean."""

        return self.loc + self.a * self.scale

    @property
    def median(self) -> Any:
        """Return the distribution median."""

        value = self.ppf(0.5)
        return float(value) if np.ndim(value) == 0 else value

    def logpdf(self, x: Any, norm: bool = True) -> np.ndarray:
        """Evaluate the log-probability density.

        Parameters
        ----------
        x : array-like
            Evaluation points.
        norm : bool, default=True
            Include the normalization constant when true.

        Returns
        -------
        numpy.ndarray
            Log-density at each evaluation point.
        """

        y = (np.asarray(x) - self.loc) * self._inverse_scale
        value = xlogy(self.a - 1.0, y) - y
        if norm:
            value = value + self._log_normalization
        return np.where(y >= 0.0, value, -np.inf)

    def pdf(self, x: Any, norm: bool = True) -> np.ndarray:
        """Evaluate the probability density.

        Parameters
        ----------
        x : array-like
            Evaluation points.
        norm : bool, default=True
            Include the normalization constant when true.

        Returns
        -------
        numpy.ndarray
            Probability density at each evaluation point.
        """

        return np.exp(self.logpdf(x, norm=norm))

    def cdf(self, x: Any) -> np.ndarray:
        """Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : array-like
            Evaluation points.

        Returns
        -------
        numpy.ndarray
            Cumulative probability at each evaluation point.
        """

        y = (np.asarray(x) - self.loc) * self._inverse_scale
        return np.where(y > 0.0, gammainc(self.a, y), 0.0)

    def _sf(self, x: Any) -> np.ndarray:
        y = (np.asarray(x) - self.loc) * self._inverse_scale
        return np.where(y > 0.0, gammaincc(self.a, y), 1.0)

    def ppf(self, q: Any) -> np.ndarray:
        """Evaluate the quantile function.

        Parameters
        ----------
        q : array-like
            Cumulative probabilities in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray
            Distribution quantiles.
        """

        q, valid = _valid_quantile(q)
        value = self.loc + self.scale * gammaincinv(self.a, q)
        return np.where(valid, value, np.nan)


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
        cdf_low = float(ndtr((low - loc) / scale))
        cdf_high = float(ndtr((high - loc) / scale))
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

    def logpdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        z = (x - self.loc) / self.scale
        return np.where(
            (x >= self.low) & (x <= self.high),
            -0.5 * z * z + self._log_normalization,
            -np.inf,
        )

    def pdf(self, x: Any) -> np.ndarray:
        return np.exp(self.logpdf(x))

    def cdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        value = ndtr((x - self.loc) / self.scale) - self._cdf_low
        value /= self._cdf_width
        return np.clip(value, 0.0, 1.0)

    def ppf(self, q: Any) -> np.ndarray:
        q, valid = _valid_quantile(q)
        probability = self._cdf_low + q * self._cdf_width
        value = self.loc + self.scale * ndtri(probability)
        value = np.where(q == 0.0, self.low, value)
        value = np.where(q == 1.0, self.high, value)
        return np.where(valid, value, np.nan)


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

    def logpdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = self._log_normalization - self.alpha * np.log(x)
        return np.where((x >= self.low) & (x <= self.high), value, -np.inf)

    def pdf(self, x: Any) -> np.ndarray:
        return np.exp(self.logpdf(x))

    def cdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        if self._log_uniform:
            with np.errstate(divide="ignore", invalid="ignore"):
                value = (np.log(x) - self._lower_term) / self._width_term
        else:
            with np.errstate(invalid="ignore"):
                value = (x**self._power - self._lower_term) / self._width_term
        return np.where(x <= self.low, 0.0, np.where(x >= self.high, 1.0, value))

    def ppf(self, q: Any) -> np.ndarray:
        q, valid = _valid_quantile(q)
        if self._log_uniform:
            value = np.exp(self._lower_term + q * self._width_term)
        else:
            value = (self._lower_term + q * self._width_term) ** (1.0 / self._power)
        return np.where(valid, value, np.nan)


@dataclass(frozen=True, slots=True)
class TruncatedSine(_TailMethods):
    """Sine density on the interval ``[0, pi / 2]``."""

    @property
    def mean(self) -> float:
        return 1.0

    @property
    def median(self) -> float:
        return math.pi / 3.0

    def pdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        return np.where((x >= 0.0) & (x <= math.pi / 2.0), np.sin(x), 0.0)

    def logpdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = np.log(np.sin(x))
        return np.where((x > 0.0) & (x <= math.pi / 2.0), value, -np.inf)

    def cdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        return np.where(
            x <= 0.0,
            0.0,
            np.where(x >= math.pi / 2.0, 1.0, 1.0 - np.cos(x)),
        )

    def ppf(self, q: Any) -> np.ndarray:
        q, valid = _valid_quantile(q)
        return np.where(valid, np.arccos(1.0 - q), np.nan)


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

    def _z(self, x: Any) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log((np.asarray(x) - self.loc) / self.scale) / self.s

    def _log_density(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        z = self._z(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = -0.5 * z * z - np.log(x - self.loc)
        value -= math.log(self.s) + 0.5 * _LOG_TWO_PI
        return np.where(x > self.loc, value, -np.inf)

    def _cumulative(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        return np.where(x > self.loc, ndtr(self._z(x)), 0.0)

    def _sf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        return np.where(x > self.loc, ndtr(-self._z(x)), 1.0)

    def _logcdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        return np.where(x > self.loc, log_ndtr(self._z(x)), -np.inf)

    def _logsf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        return np.where(x > self.loc, log_ndtr(-self._z(x)), 0.0)

    def _quantile(self, q: np.ndarray) -> np.ndarray:
        return self.loc + self.scale * np.exp(self.s * ndtri(q))


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
    def median(self) -> float:
        return float(self.ppf(0.5))

    def _log_density(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        z = (x - self.loc) / self.scale
        value = -0.5 * z * z + 0.5 * math.log(2.0 / math.pi)
        return np.where(x >= self.loc, value - math.log(self.scale), -np.inf)

    def _cumulative(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        z = (x - self.loc) / self.scale
        return np.where(x > self.loc, 2.0 * ndtr(z) - 1.0, 0.0)

    def _sf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        z = (x - self.loc) / self.scale
        return np.where(x > self.loc, 2.0 * ndtr(-z), 1.0)

    def _logsf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        z = (x - self.loc) / self.scale
        return np.where(x > self.loc, math.log(2.0) + log_ndtr(-z), 0.0)

    def _quantile(self, q: np.ndarray) -> np.ndarray:
        return self.loc + self.scale * ndtri(0.5 * (q + 1.0))


@dataclass(frozen=True, slots=True)
class StudentT(_AnalyticDistribution):
    """Student's t distribution with broadcasting NumPy methods.

    Parameters
    ----------
    df : float, default=1.0
        Positive degrees of freedom.
    loc : float, default=0.0
        Distribution location.
    scale : float, default=1.0
        Positive scale parameter.
    """

    df: float = 1.0
    loc: float = 0.0
    scale: float = 1.0
    _log_normalization: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate parameters and cache the normalization constant."""

        df = _positive(self.df, "df")
        loc = _finite(self.loc, "loc")
        scale = _positive(self.scale, "scale")
        normalization = gammaln(0.5 * (df + 1.0)) - gammaln(0.5 * df)
        normalization -= 0.5 * math.log(df * math.pi) + math.log(scale)
        object.__setattr__(self, "df", df)
        object.__setattr__(self, "loc", loc)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "_log_normalization", float(normalization))

    @property
    def mean(self) -> float:
        """Return the mean, or NaN when it is undefined."""

        return self.loc if self.df > 1.0 else math.nan

    @property
    def median(self) -> float:
        """Return the distribution median."""

        return self.loc

    def _log_density(self, x: Any) -> np.ndarray:
        """Evaluate the vectorized log-density kernel."""

        z = (np.asarray(x) - self.loc) / self.scale
        return self._log_normalization - 0.5 * (self.df + 1.0) * np.log1p(
            z * z / self.df
        )

    def _tail_probability(self, z: np.ndarray) -> np.ndarray:
        """Evaluate the smaller symmetric tail at absolute standardized values."""

        ratio = self.df / (self.df + z * z)
        return 0.5 * betainc(0.5 * self.df, 0.5, ratio)

    def _cumulative(self, x: Any) -> np.ndarray:
        """Evaluate the vectorized cumulative-distribution kernel."""

        z = (np.asarray(x) - self.loc) / self.scale
        tail = self._tail_probability(z)
        return np.where(z < 0.0, tail, 1.0 - tail)

    def _sf(self, x: Any) -> np.ndarray:
        """Evaluate the survival function directly from the smaller tail."""

        z = (np.asarray(x) - self.loc) / self.scale
        tail = self._tail_probability(z)
        return np.where(z > 0.0, tail, 1.0 - tail)

    def _logcdf(self, x: Any) -> np.ndarray:
        """Evaluate the logarithmic CDF without subtractive tail loss."""

        z = (np.asarray(x) - self.loc) / self.scale
        tail = self._tail_probability(z)
        with np.errstate(divide="ignore"):
            return np.where(z < 0.0, np.log(tail), np.log1p(-tail))

    def _logsf(self, x: Any) -> np.ndarray:
        """Evaluate the logarithmic survival function directly."""

        z = (np.asarray(x) - self.loc) / self.scale
        tail = self._tail_probability(z)
        with np.errstate(divide="ignore"):
            return np.where(z > 0.0, np.log(tail), np.log1p(-tail))

    def _quantile(self, q: np.ndarray) -> np.ndarray:
        """Evaluate the vectorized inverse-CDF kernel through Beta inversion."""

        tail = 2.0 * np.minimum(q, 1.0 - q)
        ratio = betaincinv(0.5 * self.df, 0.5, tail)
        with np.errstate(divide="ignore", invalid="ignore"):
            magnitude = np.sqrt(self.df * (1.0 / ratio - 1.0))
        value = self.loc + self.scale * np.where(q < 0.5, -magnitude, magnitude)
        return np.where(q == 0.5, self.loc, value)


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

    def _log_density(self, x: Any) -> np.ndarray:
        z = (np.asarray(x) - self.loc) / self.scale
        return -math.log(math.pi * self.scale) - np.log1p(z * z)

    def _cumulative(self, x: Any) -> np.ndarray:
        z = (np.asarray(x) - self.loc) / self.scale
        return 0.5 + np.arctan(z) / math.pi

    def _sf(self, x: Any) -> np.ndarray:
        z = (np.asarray(x) - self.loc) / self.scale
        reciprocal = np.divide(1.0, z, out=np.zeros_like(z), where=z != 0.0)
        positive = np.arctan(reciprocal) / math.pi
        return np.where(z > 0.0, positive, 0.5 - np.arctan(z) / math.pi)

    def _quantile(self, q: np.ndarray) -> np.ndarray:
        value = self.loc + self.scale * np.tan(math.pi * (q - 0.5))
        value = np.where(q == 0.0, -np.inf, value)
        return np.where(q == 1.0, np.inf, value)


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

    def _log_density(self, x: Any) -> np.ndarray:
        value = -np.abs(np.asarray(x) - self.loc) / self.scale
        return value - math.log(2.0 * self.scale)

    def _cumulative(self, x: Any) -> np.ndarray:
        z = (np.asarray(x) - self.loc) / self.scale
        return np.where(z <= 0.0, 0.5 * np.exp(z), 1.0 - 0.5 * np.exp(-z))

    def _sf(self, x: Any) -> np.ndarray:
        z = (np.asarray(x) - self.loc) / self.scale
        return np.where(z <= 0.0, 1.0 - 0.5 * np.exp(z), 0.5 * np.exp(-z))

    def _logcdf(self, x: Any) -> np.ndarray:
        z = (np.asarray(x) - self.loc) / self.scale
        with np.errstate(invalid="ignore"):
            positive = np.log1p(-0.5 * np.exp(-z))
        return np.where(z <= 0.0, math.log(0.5) + z, positive)

    def _logsf(self, x: Any) -> np.ndarray:
        z = (np.asarray(x) - self.loc) / self.scale
        with np.errstate(invalid="ignore"):
            negative = np.log1p(-0.5 * np.exp(z))
        return np.where(z <= 0.0, negative, math.log(0.5) - z)

    def _quantile(self, q: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            lower = self.loc + self.scale * np.log(2.0 * q)
            upper = self.loc - self.scale * np.log(2.0 * (1.0 - q))
        return np.where(q < 0.5, lower, upper)


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

    def _log_density(self, x: Any) -> np.ndarray:
        z = (np.asarray(x) - self.loc) / self.scale
        with np.errstate(divide="ignore", invalid="ignore"):
            value = math.log(self.c / self.scale)
            value += (self.c - 1.0) * np.log(z) - z**self.c
        boundary = -np.inf if self.c > 1.0 else np.inf
        if self.c == 1.0:
            boundary = -math.log(self.scale)
        return np.where(z < 0.0, -np.inf, np.where(z == 0.0, boundary, value))

    def _cumulative(self, x: Any) -> np.ndarray:
        z = (np.asarray(x) - self.loc) / self.scale
        with np.errstate(invalid="ignore"):
            value = -np.expm1(-(z**self.c))
        return np.where(z > 0.0, value, 0.0)

    def _sf(self, x: Any) -> np.ndarray:
        z = (np.asarray(x) - self.loc) / self.scale
        with np.errstate(invalid="ignore"):
            value = np.exp(-(z**self.c))
        return np.where(z > 0.0, value, 1.0)

    def _logsf(self, x: Any) -> np.ndarray:
        z = (np.asarray(x) - self.loc) / self.scale
        with np.errstate(invalid="ignore"):
            value = -(z**self.c)
        return np.where(z > 0.0, value, 0.0)

    def _quantile(self, q: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return self.loc + self.scale * (-np.log1p(-q)) ** (1.0 / self.c)


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
    def median(self) -> float:
        return float(self.ppf(0.5))

    def pdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        with np.errstate(invalid="ignore"):
            valid = (
                np.isfinite(x) & (x >= self.low) & (x < self.high) & (x == np.floor(x))
            )
        return np.where(valid, self._mass, 0.0)

    pmf = pdf

    def logpdf(self, x: Any) -> np.ndarray:
        return np.where(self.pdf(x) > 0.0, self._log_mass, -np.inf)

    logpmf = logpdf

    def cdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        value = (np.floor(x) - self.low + 1) / self._count
        return np.clip(value, 0.0, 1.0)

    def ppf(self, q: Any) -> np.ndarray:
        q, valid = _valid_quantile(q)
        value = self.low + np.ceil(q * self._count) - 1
        value = np.where(q == 0.0, self.low, value)
        return np.where(valid, value, np.nan)


# Transitional aliases mirror the scalar and PBjam naming schemes.
normal = Normal
uniform = Uniform
beta = Beta
gamma = Gamma
truncsine = TruncatedSine
randint = DiscreteUniform
