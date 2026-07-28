# Benchmark conclusions for the first pre-release

Baldr's benchmark harness separates construction, initial call or compilation,
and synchronized warm calls. The initial development sequence supports three
practical conclusions:

- scalar Python should remain the default for repeated scalar sampler calls;
- NumPy broadcasting is the appropriate CPU array path; and
- JAX is most useful when a complete multi-parameter transform or likelihood
  is compiled as one program.

Method-level JIT is retained as a comparison, not the recommended execution
model. It introduces a Python dispatch and compiled-kernel boundary for every
distribution method.

Empirical distributions follow the same separation of costs. KDE fitting and
grid normalization occur once, while warm evaluations use interpolation only.
The fitted grid can be transferred to JAX without repeating the fit.

No universal timing table is included because absolute results depend on CPU,
accelerator, dtype, JAX compilation cache, and dependency versions. The JSON
benchmark output records those details and is the reproducible unit of
comparison. See [the benchmarking policy](benchmarking.md) for commands and
interpretation rules.
