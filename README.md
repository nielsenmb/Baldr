# Baldr

Baldr is an experimental Python package for probability distributions that are
fast to evaluate in sampling and inference workloads.

The project is benchmark-first. It will provide deliberately lean execution
paths for three different use cases:

- dependency-free scalar evaluation for samplers such as Dynesty;
- NumPy broadcasting for CPU vector workloads; and
- JAX-traceable kernels for larger compiled transforms and likelihoods.

These paths will share a distribution-level public API, but backend selection
will happen when a distribution is constructed rather than during every
evaluation. JAX will remain optional.

Baldr is not intended to replace the validation and broad flexibility of
`scipy.stats`. Users are expected to validate parameters at model boundaries;
the hot path can then remain small.

## Status

Baldr provides scalar, NumPy, and JAX backends through a shared set of public
constructors:

```python
from baldr import Gamma, Normal

scalar_prior = Normal(loc=0.0, scale=1.0)
array_prior = Normal(loc=0.0, scale=1.0, backend="numpy")
jax_prior = Normal(loc=0.0, scale=1.0, backend="jax")
gamma_prior = Gamma(a=2.0, scale=3.0)

x = scalar_prior.ppf(0.95)
log_density = array_prior.logpdf([x, 0.0])
```

Backend selection happens once during construction, so repeated evaluations do
not pay for a backend branch. Install `baldr[numpy]` for NumPy broadcasting or
`baldr[jax]` for traceable JAX arrays. Baldr does not JIT individual methods:
compile the complete prior transform or likelihood to let JAX fuse operations.

Baldr currently provides Normal, Uniform, Beta, Gamma, Exponential, truncated
Normal, truncated power-law, truncated sine, and discrete uniform
distributions. `CallableDistribution` adapts existing `pdf`, `logpdf`, `cdf`,
and `ppf` functions without imposing a backend or performing construction-time
numerical integration.

Empirical priors can be fitted once and evaluated repeatedly through cached
interpolation grids:

```python
from baldr.empirical import fit_empirical

prior = fit_empirical(samples, backend="numpy")
jax_prior = prior.grid.to_backend("jax")  # reuses the fitted KDE grid
```

Install `baldr[empirical]` to fit empirical distributions. JAX evaluation also
requires `baldr[jax]`.

The first benchmark uses the Normal log-PDF as a control and separates
construction, first-call or compilation, and warm-call timings. See
[the benchmarking policy](docs/benchmarking.md) for the comparison set and
instructions.

The staged implementation is described in the
[development plan](docs/development-plan.md).
Users moving existing priors can follow the
[PBjam and AsteroScale migration guide](docs/migration.md).
The [API reference](docs/api.md) defines the supported pre-release surface, and
the [empirical-prior guide](docs/empirical.md) explains the approximation and
backend tradeoffs.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```
