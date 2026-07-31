"""Dependency-free scalar probability distributions.

The classes in this module validate parameters once during construction and
then use only Python's :mod:`math` and :mod:`statistics` modules in their hot
methods.  They intentionally accept scalar values only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

__all__ = [
    "Beta",
    "Cauchy",
    "Gamma",
    "DiscreteUniform",
    "Exponential",
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
_STANDARD_NORMAL = NormalDist()


class _TailMethods:
    """Provide survival and logarithmic cumulative methods."""

    def sf(self, x: float) -> float:
        """Evaluate the survival function.

        Parameters
        ----------
        x : float
            Evaluation point.

        Returns
        -------
        float
            Upper-tail probability at ``x``.
        """

        survival = getattr(self, "_sf", None)
        if survival is not None:
            return survival(x)
        return max(0.0, 1.0 - self.cdf(x))

    def logcdf(self, x: float) -> float:
        """Evaluate the logarithm of the cumulative distribution function.

        Parameters
        ----------
        x : float
            Evaluation point.

        Returns
        -------
        float
            Logarithmic lower-tail probability at ``x``.
        """

        logarithm = getattr(self, "_logcdf", None)
        value = logarithm(x) if logarithm is not None else self.cdf(x)
        return value if logarithm is not None else _log_probability(value)

    def logsf(self, x: float) -> float:
        """Evaluate the logarithm of the survival function.

        Parameters
        ----------
        x : float
            Evaluation point.

        Returns
        -------
        float
            Logarithmic upper-tail probability at ``x``.
        """

        logarithm = getattr(self, "_logsf", None)
        value = logarithm(x) if logarithm is not None else self.sf(x)
        return value if logarithm is not None else _log_probability(value)


class _AnalyticDistribution(_TailMethods):
    """Expose the common scalar API for analytic distributions."""

    def logpdf(self, x: float) -> float:
        """Evaluate the log-probability density.

        Parameters
        ----------
        x : float
            Evaluation point.

        Returns
        -------
        float
            Log-density at ``x``.
        """

        return self._log_density(x)

    def pdf(self, x: float) -> float:
        """Evaluate the probability density.

        Parameters
        ----------
        x : float
            Evaluation point.

        Returns
        -------
        float
            Probability density at ``x``.
        """

        return math.exp(self._log_density(x))

    def cdf(self, x: float) -> float:
        """Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : float
            Evaluation point.

        Returns
        -------
        float
            Lower-tail probability at ``x``.
        """

        return self._cumulative(x)

    def ppf(self, q: float) -> float:
        """Evaluate the quantile function.

        Parameters
        ----------
        q : float
            Cumulative probability in ``[0, 1]``.

        Returns
        -------
        float
            Distribution quantile.
        """

        return self._quantile(q)


def _log_probability(value: float) -> float:
    """Return a boundary-safe logarithm of a probability."""

    return math.log(value) if value > 0.0 else -math.inf


def _standard_normal_logcdf(x: float) -> float:
    """Evaluate the standard Normal log-CDF without lower-tail underflow."""

    if x > -10.0:
        return math.log(0.5 * math.erfc(-x / math.sqrt(2.0)))
    inverse_square = 1.0 / (x * x)
    correction = 1.0 - inverse_square + 3.0 * inverse_square**2
    correction -= 15.0 * inverse_square**3
    return -0.5 * x * x - math.log(-x) - 0.5 * _LOG_TWO_PI + math.log(correction)


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


def _quantile_endpoint(q: float, low: float, high: float) -> float | None:
    if q == 0.0:
        return low
    if q == 1.0:
        return high
    if q < 0.0 or q > 1.0 or math.isnan(q):
        return math.nan
    return None


@dataclass(frozen=True, slots=True)
class Normal(_TailMethods):
    """Normal distribution with scalar evaluation methods."""

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

    def logpdf(self, x: float, norm: bool = True) -> float:
        z = (x - self.loc) * self._inverse_scale
        value = -0.5 * z * z
        return value + self._log_normalization if norm else value

    def pdf(self, x: float, norm: bool = True) -> float:
        value = math.exp(-0.5 * ((x - self.loc) * self._inverse_scale) ** 2)
        return value * math.exp(self._log_normalization) if norm else value

    def cdf(self, x: float) -> float:
        return _STANDARD_NORMAL.cdf((x - self.loc) * self._inverse_scale)

    def _sf(self, x: float) -> float:
        z = (x - self.loc) * self._inverse_scale
        return _STANDARD_NORMAL.cdf(-z)

    def _logcdf(self, x: float) -> float:
        return _standard_normal_logcdf((x - self.loc) * self._inverse_scale)

    def _logsf(self, x: float) -> float:
        return _standard_normal_logcdf(-(x - self.loc) * self._inverse_scale)

    def ppf(self, q: float) -> float:
        endpoint = _quantile_endpoint(q, -math.inf, math.inf)
        if endpoint is not None:
            return endpoint
        return self.loc + self.scale * _STANDARD_NORMAL.inv_cdf(q)


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

    def pdf(self, x: float) -> float:
        return self._density if self.loc <= x <= self.high else 0.0

    def logpdf(self, x: float) -> float:
        return self._log_density if self.loc <= x <= self.high else -math.inf

    def cdf(self, x: float) -> float:
        if x <= self.loc:
            return 0.0
        if x >= self.high:
            return 1.0
        return (x - self.loc) / self.scale

    def _sf(self, x: float) -> float:
        if x <= self.loc:
            return 1.0
        if x >= self.high:
            return 0.0
        return (self.high - x) / self.scale

    def ppf(self, q: float) -> float:
        endpoint = _quantile_endpoint(q, self.loc, self.high)
        if endpoint is not None:
            return endpoint
        return self.loc + q * self.scale


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete-beta continued fraction."""

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    result = d
    for iteration in range(1, 201):
        m2 = 2 * iteration
        coefficient = iteration * (b - iteration) * x
        coefficient /= (qam + m2) * (a + m2)
        d = 1.0 + coefficient * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + coefficient / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        result *= d * c

        coefficient = -(a + iteration) * (qab + iteration) * x
        coefficient /= (a + m2) * (qap + m2)
        d = 1.0 + coefficient * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + coefficient / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= 3e-14:
            break
    return result


def _regularized_beta(x: float, a: float, b: float, log_beta: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _regularized_gamma(a: float, x: float) -> float:
    """Evaluate the regularized lower incomplete gamma function.

    Parameters
    ----------
    a : float
        Positive shape parameter.
    x : float
        Non-negative evaluation point.

    Returns
    -------
    float
        Value of the regularized lower incomplete gamma function.
    """

    if x <= 0.0:
        return 0.0
    log_front = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        denominator = a
        for _ in range(1, 201):
            denominator += 1.0
            term *= x / denominator
            total += term
            if abs(term) <= abs(total) * 3e-14:
                break
        return total * math.exp(log_front)

    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / max(abs(b), tiny)
    if b < 0.0:
        d = -d
    result = d
    for iteration in range(1, 201):
        coefficient = -iteration * (iteration - a)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= 3e-14:
            break
    return 1.0 - math.exp(log_front) * result


def _regularized_gamma_upper(a: float, x: float) -> float:
    """Evaluate the regularized upper incomplete gamma function.

    Parameters
    ----------
    a : float
        Positive shape parameter.
    x : float
        Non-negative evaluation point.

    Returns
    -------
    float
        Value of the regularized upper incomplete gamma function.
    """

    if x <= 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _regularized_gamma(a, x)

    log_front = a * math.log(x) - x - math.lgamma(a)
    tiny = 1e-300
    denominator = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / max(abs(denominator), tiny)
    if denominator < 0.0:
        d = -d
    result = d
    for iteration in range(1, 201):
        coefficient = -iteration * (iteration - a)
        denominator += 2.0
        d = coefficient * d + denominator
        if abs(d) < tiny:
            d = tiny
        c = denominator + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= 3e-14:
            break
    return math.exp(log_front) * result


def _log_power_at_boundary(exponent: float) -> float:
    if exponent > 0.0:
        return -math.inf
    if exponent < 0.0:
        return math.inf
    return 0.0


@dataclass(frozen=True, slots=True)
class Beta(_TailMethods):
    """Beta distribution transformed to ``[loc, loc + scale]``."""

    a: float = 1.0
    b: float = 1.0
    loc: float = 0.0
    scale: float = 1.0
    high: float = field(init=False)
    _log_beta: float = field(init=False, repr=False)
    _log_normalization: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        a = _positive(self.a, "a")
        b = _positive(self.b, "b")
        loc = _finite(self.loc, "loc")
        scale = _positive(self.scale, "scale")
        log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "loc", loc)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "high", loc + scale)
        object.__setattr__(self, "_log_beta", log_beta)
        object.__setattr__(self, "_log_normalization", -log_beta - math.log(scale))

    @property
    def mean(self) -> float:
        return self.loc + self.scale * self.a / (self.a + self.b)

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    def logpdf(self, x: float, norm: bool = True) -> float:
        y = (x - self.loc) / self.scale
        if y < 0.0 or y > 1.0:
            return -math.inf
        left = (
            _log_power_at_boundary(self.a - 1.0)
            if y == 0.0
            else (self.a - 1.0) * math.log(y)
        )
        right = (
            _log_power_at_boundary(self.b - 1.0)
            if y == 1.0
            else (self.b - 1.0) * math.log1p(-y)
        )
        value = left + right
        return value + self._log_normalization if norm else value

    def pdf(self, x: float, norm: bool = True) -> float:
        return math.exp(self.logpdf(x, norm=norm))

    def cdf(self, x: float) -> float:
        y = (x - self.loc) / self.scale
        return _regularized_beta(y, self.a, self.b, self._log_beta)

    def _sf(self, x: float) -> float:
        y = (x - self.loc) / self.scale
        return _regularized_beta(1.0 - y, self.b, self.a, self._log_beta)

    def ppf(self, q: float) -> float:
        endpoint = _quantile_endpoint(q, self.loc, self.high)
        if endpoint is not None:
            return endpoint

        low = 0.0
        high = 1.0
        x = self.a / (self.a + self.b)
        tolerance = max(2e-14, 2e-12 * min(q, 1.0 - q))
        for _ in range(100):
            value = _regularized_beta(x, self.a, self.b, self._log_beta)
            error = value - q
            if abs(error) <= tolerance:
                break
            if value < q:
                low = x
            else:
                high = x
            log_density = self._log_normalization + math.log(self.scale)
            log_density += (self.a - 1.0) * math.log(x)
            log_density += (self.b - 1.0) * math.log1p(-x)
            density = math.exp(log_density)
            proposal = x - error / density if density > 0.0 else math.nan
            if not low < proposal < high or proposal == x:
                proposal = 0.5 * (low + high)
            if high - low <= 2e-15:
                x = proposal
                break
            x = proposal
        return self.loc + self.scale * x


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

    def logpdf(self, x: float) -> float:
        return -x * self._inverse_scale - self._log_scale if x >= 0.0 else -math.inf

    def pdf(self, x: float) -> float:
        return math.exp(self.logpdf(x))

    def cdf(self, x: float) -> float:
        return -math.expm1(-x * self._inverse_scale) if x > 0.0 else 0.0

    def _sf(self, x: float) -> float:
        return math.exp(-x * self._inverse_scale) if x >= 0.0 else 1.0

    def _logsf(self, x: float) -> float:
        return -x * self._inverse_scale if x >= 0.0 else 0.0

    def ppf(self, q: float) -> float:
        endpoint = _quantile_endpoint(q, 0.0, math.inf)
        if endpoint is not None:
            return endpoint
        return -self.scale * math.log1p(-q)


@dataclass(frozen=True, slots=True)
class Gamma(_TailMethods):
    """Gamma distribution using SciPy's shape, location, and scale convention.

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
        """Validate parameters and cache scalar normalization terms."""

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
    def median(self) -> float:
        """Return the distribution median."""

        return self.ppf(0.5)

    def logpdf(self, x: float, norm: bool = True) -> float:
        """Evaluate the log-probability density.

        Parameters
        ----------
        x : float
            Evaluation point.
        norm : bool, default=True
            Include the normalization constant when true.

        Returns
        -------
        float
            Log-density at ``x``.
        """

        y = (x - self.loc) * self._inverse_scale
        if y < 0.0:
            return -math.inf
        power = (
            _log_power_at_boundary(self.a - 1.0)
            if y == 0.0
            else ((self.a - 1.0) * math.log(y))
        )
        value = power - y
        return value + self._log_normalization if norm else value

    def pdf(self, x: float, norm: bool = True) -> float:
        """Evaluate the probability density.

        Parameters
        ----------
        x : float
            Evaluation point.
        norm : bool, default=True
            Include the normalization constant when true.

        Returns
        -------
        float
            Probability density at ``x``.
        """

        return math.exp(self.logpdf(x, norm=norm))

    def cdf(self, x: float) -> float:
        """Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : float
            Evaluation point.

        Returns
        -------
        float
            Cumulative probability at ``x``.
        """

        return _regularized_gamma(self.a, (x - self.loc) * self._inverse_scale)

    def _sf(self, x: float) -> float:
        y = (x - self.loc) * self._inverse_scale
        return _regularized_gamma_upper(self.a, y)

    def ppf(self, q: float) -> float:
        """Evaluate the quantile function.

        Parameters
        ----------
        q : float
            Cumulative probability in ``[0, 1]``.

        Returns
        -------
        float
            Distribution quantile.
        """

        endpoint = _quantile_endpoint(q, self.loc, math.inf)
        if endpoint is not None:
            return endpoint

        low = 0.0
        high = max(1.0, self.a)
        while _regularized_gamma(self.a, high) < q:
            high *= 2.0
        for _ in range(96):
            middle = 0.5 * (low + high)
            if _regularized_gamma(self.a, middle) < q:
                low = middle
            else:
                high = middle
        return self.loc + self.scale * 0.5 * (low + high)


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
        cdf_low = _STANDARD_NORMAL.cdf((low - loc) / scale)
        cdf_high = _STANDARD_NORMAL.cdf((high - loc) / scale)
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

    def logpdf(self, x: float) -> float:
        if x < self.low or x > self.high:
            return -math.inf
        z = (x - self.loc) / self.scale
        return -0.5 * z * z + self._log_normalization

    def pdf(self, x: float) -> float:
        return math.exp(self.logpdf(x))

    def cdf(self, x: float) -> float:
        if x <= self.low:
            return 0.0
        if x >= self.high:
            return 1.0
        base = _STANDARD_NORMAL.cdf((x - self.loc) / self.scale)
        return (base - self._cdf_low) / self._cdf_width

    def ppf(self, q: float) -> float:
        endpoint = _quantile_endpoint(q, self.low, self.high)
        if endpoint is not None:
            return endpoint
        probability = self._cdf_low + q * self._cdf_width
        return self.loc + self.scale * _STANDARD_NORMAL.inv_cdf(probability)


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

    def logpdf(self, x: float) -> float:
        if x < self.low or x > self.high:
            return -math.inf
        return self._log_normalization - self.alpha * math.log(x)

    def pdf(self, x: float) -> float:
        return math.exp(self.logpdf(x))

    def cdf(self, x: float) -> float:
        if x <= self.low:
            return 0.0
        if x >= self.high:
            return 1.0
        if self._log_uniform:
            return (math.log(x) - self._lower_term) / self._width_term
        return (x**self._power - self._lower_term) / self._width_term

    def ppf(self, q: float) -> float:
        endpoint = _quantile_endpoint(q, self.low, self.high)
        if endpoint is not None:
            return endpoint
        if self._log_uniform:
            return math.exp(self._lower_term + q * self._width_term)
        return (self._lower_term + q * self._width_term) ** (1.0 / self._power)


@dataclass(frozen=True, slots=True)
class TruncatedSine(_TailMethods):
    """Sine density on the interval ``[0, pi / 2]``."""

    @property
    def mean(self) -> float:
        return 1.0

    @property
    def median(self) -> float:
        return math.pi / 3.0

    def pdf(self, x: float) -> float:
        return math.sin(x) if 0.0 <= x <= math.pi / 2.0 else 0.0

    def logpdf(self, x: float) -> float:
        if x <= 0.0 or x > math.pi / 2.0:
            return -math.inf
        return math.log(math.sin(x))

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        if x >= math.pi / 2.0:
            return 1.0
        return 1.0 - math.cos(x)

    def ppf(self, q: float) -> float:
        endpoint = _quantile_endpoint(q, 0.0, math.pi / 2.0)
        if endpoint is not None:
            return endpoint
        return math.acos(1.0 - q)


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

    def _z(self, x: float) -> float:
        return math.log((x - self.loc) / self.scale) / self.s

    def _log_density(self, x: float) -> float:
        if x <= self.loc:
            return -math.inf
        z = self._z(x)
        return (
            -0.5 * z * z - math.log(x - self.loc) - math.log(self.s) - 0.5 * _LOG_TWO_PI
        )

    def _cumulative(self, x: float) -> float:
        return 0.0 if x <= self.loc else _STANDARD_NORMAL.cdf(self._z(x))

    def _sf(self, x: float) -> float:
        return 1.0 if x <= self.loc else _STANDARD_NORMAL.cdf(-self._z(x))

    def _logcdf(self, x: float) -> float:
        return -math.inf if x <= self.loc else _standard_normal_logcdf(self._z(x))

    def _logsf(self, x: float) -> float:
        return 0.0 if x <= self.loc else _standard_normal_logcdf(-self._z(x))

    def _quantile(self, q: float) -> float:
        endpoint = _quantile_endpoint(q, self.loc, math.inf)
        if endpoint is not None:
            return endpoint
        return self.loc + self.scale * math.exp(self.s * _STANDARD_NORMAL.inv_cdf(q))


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
        return self._quantile(0.5)

    def _log_density(self, x: float) -> float:
        if x < self.loc:
            return -math.inf
        z = (x - self.loc) / self.scale
        return -0.5 * z * z + 0.5 * math.log(2.0 / math.pi) - math.log(self.scale)

    def _cumulative(self, x: float) -> float:
        if x <= self.loc:
            return 0.0
        return math.erf((x - self.loc) / (self.scale * math.sqrt(2.0)))

    def _sf(self, x: float) -> float:
        if x <= self.loc:
            return 1.0
        z = (x - self.loc) / self.scale
        return 2.0 * _STANDARD_NORMAL.cdf(-z)

    def _logsf(self, x: float) -> float:
        if x <= self.loc:
            return 0.0
        z = (x - self.loc) / self.scale
        return math.log(2.0) + _standard_normal_logcdf(-z)

    def _quantile(self, q: float) -> float:
        endpoint = _quantile_endpoint(q, self.loc, math.inf)
        if endpoint is not None:
            return endpoint
        return self.loc + self.scale * _STANDARD_NORMAL.inv_cdf(0.5 * (q + 1.0))


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

    def _log_density(self, x: float) -> float:
        z = (x - self.loc) / self.scale
        return -math.log(math.pi * self.scale) - math.log1p(z * z)

    def _cumulative(self, x: float) -> float:
        z = (x - self.loc) / self.scale
        return 0.5 + math.atan(z) / math.pi

    def _sf(self, x: float) -> float:
        z = (x - self.loc) / self.scale
        if z > 0.0:
            return math.atan(1.0 / z) / math.pi
        return 0.5 - math.atan(z) / math.pi

    def _quantile(self, q: float) -> float:
        endpoint = _quantile_endpoint(q, -math.inf, math.inf)
        if endpoint is not None:
            return endpoint
        return self.loc + self.scale * math.tan(math.pi * (q - 0.5))


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

    def _log_density(self, x: float) -> float:
        return -abs(x - self.loc) / self.scale - math.log(2.0 * self.scale)

    def _cumulative(self, x: float) -> float:
        z = (x - self.loc) / self.scale
        return 0.5 * math.exp(z) if z <= 0.0 else 1.0 - 0.5 * math.exp(-z)

    def _sf(self, x: float) -> float:
        z = (x - self.loc) / self.scale
        return 1.0 - 0.5 * math.exp(z) if z <= 0.0 else 0.5 * math.exp(-z)

    def _logcdf(self, x: float) -> float:
        z = (x - self.loc) / self.scale
        return math.log(0.5) + z if z <= 0.0 else math.log1p(-0.5 * math.exp(-z))

    def _logsf(self, x: float) -> float:
        z = (x - self.loc) / self.scale
        return math.log1p(-0.5 * math.exp(z)) if z <= 0.0 else math.log(0.5) - z

    def _quantile(self, q: float) -> float:
        endpoint = _quantile_endpoint(q, -math.inf, math.inf)
        if endpoint is not None:
            return endpoint
        if q < 0.5:
            return self.loc + self.scale * math.log(2.0 * q)
        return self.loc - self.scale * math.log(2.0 * (1.0 - q))


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

    def _log_density(self, x: float) -> float:
        z = (x - self.loc) / self.scale
        if z < 0.0:
            return -math.inf
        if z == 0.0:
            return _log_power_at_boundary(self.c - 1.0)
        return math.log(self.c / self.scale) + (self.c - 1.0) * math.log(z) - z**self.c

    def _cumulative(self, x: float) -> float:
        z = (x - self.loc) / self.scale
        return 0.0 if z <= 0.0 else -math.expm1(-(z**self.c))

    def _sf(self, x: float) -> float:
        z = (x - self.loc) / self.scale
        return 1.0 if z <= 0.0 else math.exp(-(z**self.c))

    def _logsf(self, x: float) -> float:
        z = (x - self.loc) / self.scale
        return 0.0 if z <= 0.0 else -(z**self.c)

    def _quantile(self, q: float) -> float:
        endpoint = _quantile_endpoint(q, self.loc, math.inf)
        if endpoint is not None:
            return endpoint
        return self.loc + self.scale * (-math.log1p(-q)) ** (1.0 / self.c)


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
        return self.ppf(0.5)

    def pdf(self, x: float) -> float:
        return self._mass if self.low <= x < self.high and x == math.floor(x) else 0.0

    pmf = pdf

    def logpdf(self, x: float) -> float:
        return self._log_mass if self.pdf(x) else -math.inf

    logpmf = logpdf

    def cdf(self, x: float) -> float:
        if x < self.low:
            return 0.0
        if x >= self.high - 1:
            return 1.0
        return (math.floor(x) - self.low + 1) / self._count

    def ppf(self, q: float) -> int | float:
        endpoint = _quantile_endpoint(q, float(self.low), float(self.high - 1))
        if endpoint is not None:
            return int(endpoint) if math.isfinite(endpoint) else endpoint
        return self.low + math.ceil(q * self._count) - 1


# Transitional aliases ease comparisons with PBjam's lower-case class names.
normal = Normal
uniform = Uniform
beta = Beta
gamma = Gamma
truncsine = TruncatedSine
randint = DiscreteUniform
