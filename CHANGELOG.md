# Changelog

## Unreleased

- Add LogNormal, HalfNormal, Cauchy, Laplace, and Weibull distributions to the
  scalar, NumPy, and JAX backends.
- Add `sf`, `logcdf`, and `logsf` to the common distribution API, with direct
  complementary calculations for numerically sensitive tails.
- Preserve SciPy-compatible parameter conventions for the new distributions.

## 0.1.0rc1

- Add dependency-free scalar distributions for sampler hot paths.
- Add broadcasting NumPy and traceable JAX backends.
- Add construction-time backend selection and fused JAX benchmarks.
- Add Gamma, callable wrappers, and PBjam compatibility aliases.
- Add fitted empirical KDE distributions with reusable NumPy/JAX grids.
- Document the supported API, migration paths, benchmarks, and empirical
  approximation.
