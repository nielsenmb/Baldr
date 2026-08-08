# Supported API

Baldr `0.1.0` treats the names below as its supported API. Behaviour may still
change before `1.0`, but changes will be recorded in the changelog.

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

## Gamma parameter broadcasting

The NumPy Gamma backend accepts array-valued `a` and `scale` parameters. They
follow NumPy broadcasting rules when evaluated by `pdf`, `logpdf`, `cdf`, `sf`,
`logcdf`, `logsf`, and `ppf`. The fast path deliberately does not validate
parameters or mask values outside the support: callers must provide finite
positive shapes, scales, and density evaluation points.

`Gamma.logpdf_mean(x, mean)` evaluates the equivalent parameterization
`scale = mean / a` without materializing the scale array. This fused path is
intended for large periodogram likelihoods such as Skuld's and likewise assumes
positive, prevalidated inputs.

The dependency-free scalar and JAX Gamma backends currently require scalar
parameters. In particular, the JAX inverse-CDF solver treats the shape as a
static, non-differentiated construction argument.

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
details. Validation varies by backend and distribution; performance-oriented
NumPy kernels may trust prevalidated inputs and do not provide SciPy's support
masking or full runtime validation.

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
