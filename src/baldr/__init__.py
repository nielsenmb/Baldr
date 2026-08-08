"""Fast probability distributions for sampling and inference."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("baldr")
except PackageNotFoundError:
    __version__ = "0.1.0"

from baldr import scalar
from baldr.callable import CallableDistribution, distribution
from baldr.distributions import (
    Backend,
    Beta,
    Cauchy,
    DiscreteUniform,
    Exponential,
    Gamma,
    HalfNormal,
    Laplace,
    LogNormal,
    Normal,
    StudentT,
    TruncatedNormal,
    TruncatedPowerLaw,
    TruncatedSine,
    Uniform,
    Weibull,
)

__all__ = [
    "Backend",
    "Beta",
    "Cauchy",
    "CallableDistribution",
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
    "__version__",
    "distribution",
    "scalar",
]
