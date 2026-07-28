"""Fast probability distributions for sampling and inference."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("baldr")
except PackageNotFoundError:
    __version__ = "0.1.0rc1"

from baldr import scalar
from baldr.callable import CallableDistribution, distribution
from baldr.distributions import (
    Backend,
    Beta,
    DiscreteUniform,
    Exponential,
    Gamma,
    Normal,
    TruncatedNormal,
    TruncatedPowerLaw,
    TruncatedSine,
    Uniform,
)

__all__ = [
    "Backend",
    "Beta",
    "CallableDistribution",
    "DiscreteUniform",
    "Exponential",
    "Gamma",
    "Normal",
    "TruncatedNormal",
    "TruncatedPowerLaw",
    "TruncatedSine",
    "Uniform",
    "__version__",
    "distribution",
    "scalar",
]
