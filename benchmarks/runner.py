"""Benchmark runner with machine-readable output."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from benchmarks.implementations import IMPLEMENTATIONS, BenchmarkImplementation


def _synchronize(value: Any) -> Any:
    """Wait for an asynchronous array result, when supported."""

    blocker = getattr(value, "block_until_ready", None)
    return blocker() if blocker is not None else value


def _time_calls(
    function: Callable[[Any], Any],
    argument: Any,
    *,
    repeat: int,
    number: int,
    synchronize: bool,
) -> list[float]:
    """Return per-call timings in seconds."""

    samples = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        result = None
        for _ in range(number):
            result = function(argument)
            if synchronize:
                _synchronize(result)
        elapsed = time.perf_counter_ns() - start
        samples.append(elapsed / number / 1e9)
    return samples


def _array(size: int, backend: str, dtype: str) -> Any:
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.linspace(-4.0, 4.0, size, dtype=dtype)

    import numpy as np

    return np.linspace(-4.0, 4.0, size, dtype=dtype)


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("baldr", "numpy", "scipy", "jax", "numpyro", "distrax"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
    }


def benchmark(
    implementations: Sequence[BenchmarkImplementation] = IMPLEMENTATIONS,
    *,
    sizes: Sequence[int] = (1, 8, 32, 256, 4096),
    repeat: int = 7,
    number: int = 1000,
    dtype: str = "float64",
) -> dict[str, Any]:
    """Run the Normal log-PDF benchmark suite.

    Compilation or first-call time is kept separate from steady-state timings.
    Missing optional dependencies are reported rather than treated as failures.
    """

    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")

    try:
        import jax

        jax.config.update("jax_enable_x64", dtype == "float64")
    except ImportError:
        pass

    report: dict[str, Any] = {
        "metadata": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "repeat": repeat,
            "number": number,
            "dtype": dtype,
        },
        "results": [],
        "skipped": [],
    }

    for implementation in implementations:
        construction_start = time.perf_counter_ns()
        try:
            function = implementation.factory()
        except ImportError as error:
            report["skipped"].append(
                {"implementation": implementation.name, "reason": str(error)}
            )
            continue
        construction_seconds = (time.perf_counter_ns() - construction_start) / 1e9

        for size in sizes:
            is_scalar = size == 1
            if is_scalar and not implementation.supports_scalar:
                continue
            if not is_scalar and not implementation.supports_array:
                continue

            argument = (
                0.25
                if is_scalar
                else _array(size, implementation.backend, dtype=dtype)
            )

            first_call_start = time.perf_counter_ns()
            first_result = function(argument)
            if implementation.asynchronous:
                _synchronize(first_result)
            first_call_seconds = (time.perf_counter_ns() - first_call_start) / 1e9

            samples = _time_calls(
                function,
                argument,
                repeat=repeat,
                number=number,
                synchronize=implementation.asynchronous,
            )
            result = {
                "implementation": implementation.name,
                "backend": implementation.backend,
                "size": size,
                "construction_seconds": construction_seconds,
                "first_call_seconds": first_call_seconds,
                **_summary(samples),
            }
            if not all(
                math.isfinite(value)
                for key, value in result.items()
                if key.endswith("_seconds")
            ):
                message = f"Non-finite timing recorded for {implementation.name}"
                raise RuntimeError(message)
            report["results"].append(result)

    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        default="1,8,32,256,4096",
        help="Comma-separated input sizes; size 1 is the scalar benchmark.",
    )
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--number", type=int, default=1000)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the full JSON report to this path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run benchmarks from the command line."""

    arguments = _parser().parse_args(argv)
    sizes = tuple(int(value) for value in arguments.sizes.split(","))
    report = benchmark(
        sizes=sizes,
        repeat=arguments.repeat,
        number=arguments.number,
        dtype=arguments.dtype,
    )
    encoded = json.dumps(report, indent=2)
    if arguments.output is not None:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
