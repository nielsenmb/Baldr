"""Normal log-PDF implementations used as initial benchmark controls.

These functions are experimental controls rather than Baldr's public API.
Optional third-party packages are imported lazily so that one missing
competitor does not prevent the other benchmarks from running.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class BenchmarkImplementation:
    """A lazily constructed benchmark implementation.

    Parameters
    ----------
    name
        Stable name used in reports.
    backend
        Broad execution family.
    factory
        Callable returning the function to time.
    supports_scalar
        Whether the function accepts a Python scalar.
    supports_array
        Whether the function accepts array inputs.
    asynchronous
        Whether results must be synchronized before recording a timing.
    """

    name: str
    backend: str
    factory: Callable[[], Callable[[Any], Any]]
    supports_scalar: bool
    supports_array: bool
    asynchronous: bool = False


def scalar_normal_logpdf(x: float, loc: float = 0.0, scale: float = 1.0) -> float:
    """Evaluate a Normal log-PDF using only :mod:`math`."""

    z = (x - loc) / scale
    return -0.5 * (z * z + LOG_TWO_PI) - math.log(scale)


def _scalar_factory() -> Callable[[float], float]:
    return scalar_normal_logpdf


def _numpy_factory() -> Callable[[Any], Any]:
    import numpy as np

    def logpdf(x: Any, loc: float = 0.0, scale: float = 1.0) -> Any:
        z = (x - loc) / scale
        return -0.5 * (z * z + np.log(2.0 * np.pi)) - np.log(scale)

    return logpdf


def _scipy_function_factory() -> Callable[[Any], Any]:
    import numpy as np
    from scipy.special import xlogy

    def logpdf(x: Any, loc: float = 0.0, scale: float = 1.0) -> Any:
        z = (x - loc) / scale
        return -0.5 * xlogy(1.0, 2.0 * np.pi) - np.log(scale) - 0.5 * z * z

    return logpdf


def _scipy_stats_factory() -> Callable[[Any], Any]:
    from scipy.stats import norm

    distribution = norm(loc=0.0, scale=1.0)
    return distribution.logpdf


def _scipy_new_factory() -> Callable[[Any], Any]:
    try:
        from scipy.stats import Normal
    except ImportError as error:
        message = "scipy.stats.Normal requires a recent SciPy release"
        raise ImportError(message) from error

    distribution = Normal(mu=0.0, sigma=1.0)
    return distribution.logpdf


def _jax_eager_factory() -> Callable[[Any], Any]:
    import jax.numpy as jnp

    def logpdf(x: Any, loc: float = 0.0, scale: float = 1.0) -> Any:
        z = (x - loc) / scale
        return -0.5 * (z * z + jnp.log(2.0 * jnp.pi)) - jnp.log(scale)

    return logpdf


def _jax_jit_factory() -> Callable[[Any], Any]:
    import jax
    import jax.numpy as jnp

    @jax.jit
    def logpdf(x: Any, loc: float = 0.0, scale: float = 1.0) -> Any:
        z = (x - loc) / scale
        return -0.5 * (z * z + jnp.log(2.0 * jnp.pi)) - jnp.log(scale)

    return logpdf


def _jax_scipy_factory() -> Callable[[Any], Any]:
    import jax.scipy.stats as stats

    return stats.norm.logpdf


def _numpyro_factory() -> Callable[[Any], Any]:
    import numpyro.distributions as dist

    distribution = dist.Normal(loc=0.0, scale=1.0, validate_args=False)
    return distribution.log_prob


def _distrax_factory() -> Callable[[Any], Any]:
    import distrax

    distribution = distrax.Normal(loc=0.0, scale=1.0)
    return distribution.log_prob


IMPLEMENTATIONS = (
    BenchmarkImplementation(
        "python_math",
        "scalar",
        _scalar_factory,
        supports_scalar=True,
        supports_array=False,
    ),
    BenchmarkImplementation(
        "numpy", "numpy", _numpy_factory, supports_scalar=True, supports_array=True
    ),
    BenchmarkImplementation(
        "scipy_special",
        "numpy",
        _scipy_function_factory,
        supports_scalar=True,
        supports_array=True,
    ),
    BenchmarkImplementation(
        "scipy_stats",
        "numpy",
        _scipy_stats_factory,
        supports_scalar=True,
        supports_array=True,
    ),
    BenchmarkImplementation(
        "scipy_new",
        "numpy",
        _scipy_new_factory,
        supports_scalar=True,
        supports_array=True,
    ),
    BenchmarkImplementation(
        "jax_eager",
        "jax",
        _jax_eager_factory,
        supports_scalar=True,
        supports_array=True,
        asynchronous=True,
    ),
    BenchmarkImplementation(
        "jax_jit",
        "jax",
        _jax_jit_factory,
        supports_scalar=True,
        supports_array=True,
        asynchronous=True,
    ),
    BenchmarkImplementation(
        "jax_scipy",
        "jax",
        _jax_scipy_factory,
        supports_scalar=True,
        supports_array=True,
        asynchronous=True,
    ),
    BenchmarkImplementation(
        "numpyro",
        "jax",
        _numpyro_factory,
        supports_scalar=True,
        supports_array=True,
        asynchronous=True,
    ),
    BenchmarkImplementation(
        "distrax",
        "jax",
        _distrax_factory,
        supports_scalar=True,
        supports_array=True,
        asynchronous=True,
    ),
)
