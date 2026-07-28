"""Fast probability distributions for sampling and inference."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("baldr")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

from baldr import scalar

__all__ = ["__version__", "scalar"]
