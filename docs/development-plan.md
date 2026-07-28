# Development plan

Baldr will be developed in small, benchmark-led pull requests. Each backend is
kept separate so that performance decisions can be revised without changing
the distribution-level API.

## PR 1 — Scaffold and benchmark controls

- Create the installable package and CI matrix.
- Establish machine-readable benchmark output with environment metadata.
- Separate construction, first-call/compilation, and warm-call costs.
- Compare scalar Python, NumPy, SciPy, JAX, NumPyro, and Distrax Normal
  log-PDF controls where installed.
- Exercise scalar inputs, several array sizes, and both floating-point widths.

## PR 2 — Scalar backend

- Port and correct Normal, Uniform, Beta, Exponential, Truncated Normal,
  Truncated Power Law, Truncated Sine, and Discrete Uniform.
- Provide `pdf`, `logpdf`, `cdf`, and `ppf` where mathematically appropriate.
- Add parameter validation at construction, with validation absent from hot
  methods.
- Test support boundaries, extreme probabilities, and PBjam compatibility.

The scalar backend is available experimentally as :mod:`baldr.scalar`.

## PR 3 — NumPy backend and public API

- Add broadcasting array kernels and frozen distribution objects.
- Select scalar or NumPy callables once during construction.
- Measure object, closure, and direct-function overhead before fixing the API.
- Add numerical cross-checks against SciPy and array-shape tests.

The NumPy backend is available experimentally as :mod:`baldr.numpy`. Public
constructors select the scalar or NumPy backend once, when the distribution is
created.

## PR 4 — Optional JAX backend

- Add traceable kernels without method-level JIT by default.
- Test gradients, `vmap`, whole-transform `jit`, and 32/64-bit behaviour.
- Benchmark isolated calls and realistic fused prior transforms separately.
- Document compilation-cache and static-argument pitfalls.

## PR 5 — Additional distributions and PBjam/AsteroScale migration

- Add Gamma and other generally useful functions identified during migration.
- Port the generic callable distribution without astrophysics-specific policy.
- Add compatibility guidance and update PBjam and AsteroScale consumers in
  separate downstream PRs.

## PR 6 — Empirical distributions and release hardening

- Add KDE/interpolated distributions as an optional component.
- Remove construction-time repeated work from hot workflows.
- Add API documentation, usage notebooks, a benchmark report, and release
  metadata.
- Define the supported public API and prepare the first pre-release.
