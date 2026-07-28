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

Baldr is in its benchmark and API-design phase. Scalar and NumPy backends are
available through the experimental public constructors:

```python
from baldr import Normal

scalar_prior = Normal(loc=0.0, scale=1.0)
array_prior = Normal(loc=0.0, scale=1.0, backend="numpy")

x = scalar_prior.ppf(0.95)
log_density = array_prior.logpdf([x, 0.0])
```

Backend selection happens once during construction, so repeated evaluations do
not pay for a backend branch. The scalar backend remains dependency-free. Install
`baldr[numpy]` for broadcasting over NumPy arrays; SciPy supplies the special
functions that NumPy itself does not provide.

The first benchmark uses the Normal log-PDF as a control and separates
construction, first-call or compilation, and warm-call timings. See
[the benchmarking policy](docs/benchmarking.md) for the comparison set and
instructions.

The staged implementation is described in the
[development plan](docs/development-plan.md).

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```
