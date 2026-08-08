# Changelog

## Unreleased

## 0.1.0 - 2026-08-08

- Support broadcastable array-valued Gamma shape and scale parameters in the
  NumPy backend, including density, cumulative, tail, and quantile methods.

- Add Student's t distribution to the scalar, NumPy, and JAX backends, with
  stable symmetric tails and inverse CDFs built on the hardened Beta solver.
- Replace fixed-loop JAX Beta and Gamma inverse-CDF bisection with safeguarded
  Newton solvers, direct upper-tail inversion, and quantile gradients.
- Accelerate the dependency-free scalar Gamma inverse CDF with safeguarded
  Newton updates and explicit convergence failure reporting.
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
