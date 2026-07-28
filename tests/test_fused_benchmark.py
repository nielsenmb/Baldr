"""Smoke tests for the fused JAX prior-transform benchmark."""

from baldr.benchmarks.fused import benchmark_fused_prior_transform


def test_fused_benchmark_reports_all_strategies():
    report = benchmark_fused_prior_transform(
        dimensions=7,
        repeat=1,
        number=1,
        dtype="float32",
    )
    names = {result["implementation"] for result in report["results"]}
    assert names == {
        "jax_eager_transform",
        "jax_method_jit_transform",
        "jax_fused_jit_transform",
    }
    assert all(result["first_call_seconds"] >= 0.0 for result in report["results"])
