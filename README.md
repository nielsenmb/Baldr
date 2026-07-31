# Baldr

Baldr is a small Python package for probability distributions that are fast to
evaluate in sampling and inference workloads.

It provides deliberately lean execution paths for three use cases:

- dependency-free scalar evaluation for samplers such as Dynesty;
- NumPy broadcasting for CPU vector workloads; and
- JAX-traceable kernels for larger compiled transforms and likelihoods.

These paths share a distribution-level public API. Backend selection happens
when a distribution is constructed rather than during every evaluation, and
JAX remains optional.

Baldr is not intended to replace the validation and broad flexibility of
`scipy.stats`. Users are expected to validate parameters at model boundaries
so the repeated hot path can remain small.

## Installation

Baldr is not currently published on PyPI. Install the current version directly
from GitHub:

```bash
python -m pip install "baldr @ git+https://github.com/nielsenmb/Baldr.git"
```

Add the extras required by the intended backend or workflow:

```bash
# NumPy/SciPy array backend
python -m pip install "baldr[numpy] @ git+https://github.com/nielsenmb/Baldr.git"

# JAX backend
python -m pip install "baldr[jax] @ git+https://github.com/nielsenmb/Baldr.git"

# Empirical KDE fitting
python -m pip install "baldr[empirical] @ git+https://github.com/nielsenmb/Baldr.git"

# Benchmark notebooks and comparison libraries
python -m pip install "baldr[notebook] @ git+https://github.com/nielsenmb/Baldr.git"
```

A branch, tag, or commit can be pinned by appending `@<ref>` to the repository
URL, for example `Baldr.git@main`.

## Usage

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
not pay for a backend branch. Baldr does not JIT individual methods: compile
the complete prior transform or likelihood to let JAX fuse operations.

Baldr provides Normal, Uniform, Beta, Gamma, Exponential, log-normal,
half-normal, Student's t, Cauchy, Laplace, Weibull, truncated Normal, truncated
power-law, truncated sine, and discrete uniform distributions. Distributions expose
`sf`, `logcdf`, and `logsf` in addition to `pdf`, `logpdf`, `cdf`, and `ppf`;
the direct tail methods avoid precision loss from expressions such as
`1 - cdf(x)`.
`CallableDistribution` adapts existing `pdf`, `logpdf`, `cdf`, and `ppf`
functions without imposing a backend or performing construction-time numerical
integration.

Empirical priors can be fitted once and evaluated repeatedly through cached
interpolation grids:

```python
from baldr.empirical import fit_empirical

prior = fit_empirical(samples, backend="numpy")
jax_prior = prior.grid.to_backend("jax")  # reuses the fitted KDE grid
```

## Documentation

- [Documentation overview](docs/README.md)
- [Supported API](docs/api.md)
- [Benchmarking policy and notebooks](docs/benchmarking.md)
- [Benchmark conclusions](docs/benchmark-report.md)
- [Empirical-prior guide](docs/empirical.md)
- [PBjam and AsteroScale migration guide](docs/migration.md)
- [Development history](docs/development-plan.md)
- [Changelog](CHANGELOG.md)

## Development

Clone the repository, then install it in editable mode:

```bash
git clone https://github.com/nielsenmb/Baldr.git
cd Baldr
python -m pip install -e ".[dev]"
pytest
ruff check .
```
