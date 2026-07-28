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

Baldr is in its benchmark and API-design phase. The dependency-free scalar
backend is available experimentally:

```python
from baldr.scalar import Normal

prior = Normal(loc=0.0, scale=1.0)
x = prior.ppf(0.95)
log_density = prior.logpdf(x)
```

Scalar classes validate their parameters during construction, not during
repeated density evaluations. They accept Python scalar values only; NumPy
broadcasting and the backend-selecting public API will follow in PR 3.

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
