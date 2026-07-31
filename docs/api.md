# Supported API

Baldr `0.1.0rc1` treats the names below as its supported pre-release API.
Behaviour may still change before `1.0`, but changes will be recorded in the
changelog.

## Backend-selecting constructors

The following names are importable directly from `baldr`:

- `Normal`
- `Uniform`
- `Beta`
- `Gamma`
- `Exponential`
- `LogNormal`
- `HalfNormal`
- `Cauchy`
- `StudentT`
- `Laplace`
- `Weibull`
- `TruncatedNormal`
- `TruncatedPowerLaw`
- `TruncatedSine`
- `DiscreteUniform`
- `CallableDistribution`

Except for `CallableDistribution`, each constructor accepts
`backend="scalar"`, `"numpy"`, or `"jax"`. Backend selection occurs during
construction. Every mathematical distribution implements `pdf`, `logpdf`,
`cdf`, `sf`, `logcdf`, `logsf`, and `ppf`; discrete uniform also exposes `pmf`
and `logpmf`.

The tail methods are not merely convenience aliases. Where the backend offers
a complementary special function, Baldr evaluates the upper tail directly.
For example, Normal `logsf(x)` uses a log-CDF at `-x`, and Beta and Gamma `sf`
use complementary incomplete functions. This avoids rounding a CDF to one
before subtracting it.

The lower-case names in `baldr.scalar` are transitional PBjam compatibility
aliases.

## Empirical distributions

The optional `baldr.empirical` module exports:

- `fit_empirical`
- `fit_empirical_marginals`
- `EmpiricalGrid`
- `NumPyEmpiricalDistribution`
- `JAXEmpiricalDistribution`

Fitting always requires NumPy and SciPy. Evaluation can use NumPy or JAX.
`EmpiricalGrid.to_backend` changes the evaluator without repeating the KDE fit.

## Stability boundaries

Private names beginning with `_`, benchmark internals, cached dataclass fields,
and the exact root-finding or interpolation algorithms are implementation
details. Baldr validates distribution parameters during construction but does
not provide SciPy's full runtime validation or distribution catalogue.

## Inverse-CDF solvers

The scalar and JAX Beta and Gamma inverse CDFs use safeguarded Newton updates:
Newton steps provide rapid local convergence, while a maintained bracket and
bisection fallback prevent unstable updates from leaving the support. Upper
tails are inverted through complementary special functions rather than by
subtracting probabilities from one.

JAX Beta and Gamma quantiles define derivatives with respect to cumulative
probability using ``d ppf(q) / d q = 1 / pdf(ppf(q))``. Shape parameters are
validated Python scalars selected during construction and are not
differentiable arguments of the distribution objects.
