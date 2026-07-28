"""Benchmark a realistic multi-distribution JAX prior transform."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any


def _synchronize(value: Any) -> Any:
    blocker = getattr(value, "block_until_ready", None)
    return blocker() if blocker is not None else value


def _time_calls(
    function: Callable[[Any], Any],
    argument: Any,
    *,
    repeat: int,
    number: int,
) -> dict[str, float]:
    samples = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        result = None
        for _ in range(number):
            result = function(argument)
            _synchronize(result)
        samples.append((time.perf_counter_ns() - start) / number / 1e9)
    return {
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
    }


def _transform_methods(dimensions: int):
    from baldr import (
        Beta,
        Exponential,
        Normal,
        TruncatedNormal,
        TruncatedPowerLaw,
        TruncatedSine,
        Uniform,
    )

    factories = (
        lambda: Normal(loc=0.0, scale=1.0, backend="jax").ppf,
        lambda: Uniform(loc=-2.0, scale=4.0, backend="jax").ppf,
        lambda: Beta(a=2.0, b=5.0, backend="jax").ppf,
        lambda: Exponential(scale=2.0, backend="jax").ppf,
        lambda: TruncatedNormal(
            loc=0.0, scale=1.0, low=-2.0, high=3.0, backend="jax"
        ).ppf,
        lambda: TruncatedPowerLaw(
            alpha=2.35, low=0.1, high=10.0, backend="jax"
        ).ppf,
        lambda: TruncatedSine(backend="jax").ppf,
    )
    return tuple(factories[index % len(factories)]() for index in range(dimensions))


def benchmark_fused_prior_transform(
    *,
    dimensions: int = 24,
    repeat: int = 7,
    number: int = 1000,
    dtype: str = "float64",
) -> dict[str, Any]:
    """Compare eager, method-jitted, and whole-transform-jitted execution."""

    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", dtype == "float64")
    probabilities = jnp.linspace(0.05, 0.95, dimensions, dtype=dtype)
    methods = _transform_methods(dimensions)

    def eager(values):
        transformed = [
            method(values[index]) for index, method in enumerate(methods)
        ]
        return jnp.stack(transformed)

    compiled_methods = tuple(jax.jit(method) for method in methods)

    def method_jit(values):
        return jnp.stack(
            [
                method(values[index])
                for index, method in enumerate(compiled_methods)
            ]
        )

    strategies = (
        ("jax_eager_transform", eager),
        ("jax_method_jit_transform", method_jit),
        ("jax_fused_jit_transform", jax.jit(eager)),
    )
    results = []
    for name, function in strategies:
        start = time.perf_counter_ns()
        first_result = function(probabilities)
        _synchronize(first_result)
        first_call_seconds = (time.perf_counter_ns() - start) / 1e9
        results.append(
            {
                "implementation": name,
                "backend": "jax",
                "dimensions": dimensions,
                "first_call_seconds": first_call_seconds,
                **_time_calls(
                    function,
                    probabilities,
                    repeat=repeat,
                    number=number,
                ),
            }
        )

    return {
        "dimensions": dimensions,
        "dtype": dtype,
        "repeat": repeat,
        "number": number,
        "results": results,
    }
