# Migrating PBjam and AsteroScale distributions

Baldr keeps backend selection explicit at construction. This lets scalar
samplers avoid array-conversion overhead while NumPy and JAX consumers retain
broadcasting or tracing.

## PBjam

The lower-case classes remain available from the scalar backend:

```python
from baldr.scalar import beta, normal, randint, truncsine, uniform
```

They map to `Beta`, `Normal`, `DiscreteUniform`, `TruncatedSine`, and `Uniform`,
respectively. Existing scalar `pdf`, `logpdf`, `cdf`, and `ppf` calls can
therefore migrate by changing the import. Baldr corrects the legacy
`randint.logpdf` behaviour and defines its support, like
`numpy.random.Generator.integers`, as the integers in `[low, high)`.

PBjam's generic wrapper maps to:

```python
from baldr import CallableDistribution

prior = CallableDistribution(ppf=ppf, pdf=pdf, logpdf=logpdf, cdf=cdf)
```

The transitional lower-case alias `distribution` is also available. Baldr
derives `median` from `ppf(0.5)` but deliberately leaves `mean` as `None` unless
it is supplied. This removes the legacy construction-time grid integration.
Pass an RNG exposing `uniform(0, 1)` or `random()` when reproducible `rv()`
draws are required.

## AsteroScale

AsteroScale's current array-aware classes map to Baldr's NumPy backend:

```python
from baldr import Exponential, Normal, TruncatedNormal, TruncatedPowerLaw

mass_prior = TruncatedPowerLaw(
    alpha=2.35,
    low=0.1,
    high=10.0,
    backend="numpy",
)
```

`Uniform` uses SciPy's `loc` and `scale` convention. Gamma follows the same
location and scale convention as `scipy.stats.gamma`:

```python
from baldr import Gamma

distance_component = Gamma(a=3.0, scale=length_scale, backend="numpy")
```

Astrophysics-specific combinations, such as a distance or parallax prior,
should remain in AsteroScale and compose these general distributions. Baldr
does not attach physical interpretation or parameter policy to them.

## JAX consumers

Select `backend="jax"` and JIT the complete prior transform or likelihood.
Baldr methods are traceable but are not individually decorated with
`jax.jit`. This keeps compilation boundaries under the caller's control.

## PBjam empirical priors

PBjam's `makeDistObject(data)` fits one independent KDE to every column.
Baldr provides the same data layout explicitly:

```python
from baldr.empirical import fit_empirical_marginals

priors = fit_empirical_marginals(data, backend="jax")
```

The default `normal_reference` bandwidth reproduces PBjam's robust
normal-reference rule. Baldr normalizes the finite interpolation grid and uses
the same cached grid for PDF, CDF, and PPF evaluation. Consequently, small tail
differences from PBjam's separately constructed PPF grid are expected.

For a single marginal, use `fit_empirical(sample)`. A fitted NumPy prior can be
transferred to JAX without repeating the KDE:

```python
numpy_prior = fit_empirical(sample)
jax_prior = numpy_prior.grid.to_backend("jax")
```
