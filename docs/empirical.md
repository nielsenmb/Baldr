# Empirical priors

`fit_empirical` converts a one-dimensional sample into a fast interpolated
distribution. The fit has two stages:

1. SciPy evaluates a Gaussian KDE on a finite support grid.
2. Baldr normalizes that grid and precomputes a cumulative grid for CDF and PPF
   interpolation.

The KDE is therefore a fitting cost, not an evaluation cost. Repeated
`pdf`, `logpdf`, `cdf`, and `ppf` calls only perform linear interpolation.

```python
from baldr.empirical import fit_empirical

prior = fit_empirical(
    samples,
    bandwidth="normal_reference",
    cut=5.0,
    grid_size=512,
)

theta = prior.ppf(unit_cube_value)
```

## Bandwidth and support

The default `normal_reference` rule matches the robust rule used by PBjam:
the bandwidth scale is the smaller of the sample standard deviation and the
interquartile range divided by 1.349. `scott`, `silverman`, and a positive
SciPy bandwidth factor are also accepted.

`cut` controls how many kernel bandwidths are included beyond the smallest and
largest samples. Density outside this finite support is defined as zero. The
grid is normalized after truncation, making the PDF, CDF, and PPF internally
consistent.

Increasing `grid_size` improves interpolation accuracy but increases the
constant memory footprint and JAX compilation input size. The default of 512
is intended as a practical starting point rather than a universal optimum.

## NumPy and JAX

Fitting is always a NumPy/SciPy operation. Select `backend="jax"` when the
result will be part of a larger compiled transform:

```python
import jax

jax_prior = fit_empirical(samples, backend="jax")
compiled_ppf = jax.jit(jax_prior.ppf)
```

To use both evaluators, fit once:

```python
numpy_prior = fit_empirical(samples)
jax_prior = numpy_prior.grid.to_backend("jax")
```

As with Baldr's analytic JAX distributions, compile the complete prior
transform when possible rather than each method independently.

## Independent marginals

`fit_empirical_marginals(data)` fits one distribution to each column of an
array shaped `(n_samples, n_dimensions)`. It does not model correlations
between columns; a multivariate KDE or copula is a different model.
