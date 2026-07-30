# Baldr documentation

Baldr is organized around three execution backends with one construction-time
API: scalar Python for repeated scalar calls, NumPy for CPU arrays, and JAX for
larger compiled calculations.

## Guides

- [Supported API](api.md) — public constructors, methods, and stability boundaries.
- [Benchmarking](benchmarking.md) — reproducible timing and numerical-comparison
  methodology, including the worked notebooks.
- [Benchmark conclusions](benchmark-report.md) — the practical backend choices
  supported by the current measurements.
- [Empirical priors](empirical.md) — fitted KDE grids and NumPy/JAX evaluation.
- [Migration](migration.md) — moving PBjam and AsteroScale distributions to Baldr.
- [Development history](development-plan.md) — the completed benchmark-led PR
  sequence.
- [Changelog](../CHANGELOG.md) — release-facing changes.

## Worked notebooks

- [Distribution performance](../notebooks/distribution-performance.ipynb)
- [Numerical agreement with SciPy](../notebooks/numerical-agreement.ipynb)
- [JAX library comparison](../notebooks/jax-library-comparison.ipynb)
- [Empirical priors](../notebooks/empirical-priors.ipynb)

Installation instructions are in the [project README](../README.md).
