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
from scipy.special import betainc, betaincinv, betaln, ndtr, ndtri, xlog1py, xlogy

__all__ = [
    "Beta",
    "DiscreteUniform",
    "Exponential",
    "Normal",
    "TruncatedNormal",
    "TruncatedPowerLaw",
    "TruncatedSine",
    "Uniform",
    "beta",
    "normal",
    "randint",
    "truncsine",
    "uniform",
]

_LOG_TWO_PI = math.log(2.0 * math.pi)


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
class Normal:
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

    def ppf(self, q: Any) -> np.ndarray:
        q, valid = _valid_quantile(q)
        value = self.loc + self.scale * ndtri(q)
        return np.where(valid, value, np.nan)


@dataclass(frozen=True, slots=True)
class Uniform:
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
        return np.where(
            (x >= self.loc) & (x <= self.high), self._log_density, -np.inf
        )

    def cdf(self, x: Any) -> np.ndarray:
        return np.clip((np.asarray(x) - self.loc) / self.scale, 0.0, 1.0)

    def ppf(self, q: Any) -> np.ndarray:
        q, valid = _valid_quantile(q)
        return np.where(valid, self.loc + q * self.scale, np.nan)


@dataclass(frozen=True, slots=True)
class Beta:
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

    def ppf(self, q: Any) -> np.ndarray:
        q, valid = _valid_quantile(q)
        value = self.loc + self.scale * betaincinv(self.a, self.b, q)
        return np.where(valid, value, np.nan)


@dataclass(frozen=True, slots=True)
class Exponential:
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
        return np.where(
            x >= 0.0, -x * self._inverse_scale - self._log_scale, -np.inf
        )

    def pdf(self, x: Any) -> np.ndarray:
        return np.exp(self.logpdf(x))

    def cdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        return np.where(x > 0.0, -np.expm1(-x * self._inverse_scale), 0.0)

    def ppf(self, q: Any) -> np.ndarray:
        q, valid = _valid_quantile(q)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = -self.scale * np.log1p(-q)
        return np.where(valid, value, np.nan)


@dataclass(frozen=True, slots=True)
class TruncatedNormal:
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
        value = (ndtr((x - self.loc) / self.scale) - self._cdf_low)
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
class TruncatedPowerLaw:
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
            value = (self._lower_term + q * self._width_term) ** (
                1.0 / self._power
            )
        return np.where(valid, value, np.nan)


@dataclass(frozen=True, slots=True)
class TruncatedSine:
    """Sine density on the interval ``[0, pi / 2]``."""

    @property
    def mean(self) -> float:
        return 1.0

    @property
    def median(self) -> float:
        return math.pi / 3.0

    def pdf(self, x: Any) -> np.ndarray:
        x = np.asarray(x)
        return np.where(
            (x >= 0.0) & (x <= math.pi / 2.0), np.sin(x), 0.0
        )

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
class DiscreteUniform:
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
                np.isfinite(x)
                & (x >= self.low)
                & (x < self.high)
                & (x == np.floor(x))
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
truncsine = TruncatedSine
randint = DiscreteUniform
