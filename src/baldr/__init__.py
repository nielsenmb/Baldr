"""Fast probability distributions for sampling and inference.

The public distribution API will be introduced after the benchmark phase has
established which execution paths are worth supporting.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("baldr")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

__all__ = ["__version__"]
