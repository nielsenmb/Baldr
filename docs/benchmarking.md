# Benchmarking policy

Baldr is intended for numerical hot paths. Implementation choices therefore
need measurements in the calling patterns that matter, rather than a single
headline timing.

## Initial questions

The first benchmark asks how Normal log-PDF evaluation behaves across:

- dependency-free scalar Python;
- hand-written NumPy;
- NumPy with `scipy.special`;
- a frozen `scipy.stats` distribution;
- eager and JIT-compiled JAX;
- `jax.scipy.stats`;
- SciPy's newer distribution API when the installed version provides it;
- NumPyro with argument validation disabled; and
- Distrax.

Normal is a control distribution, not the complete target API. It is simple
enough that framework overhead is visible and has implementations in every
comparison package.

## Timing model

Each result reports these costs separately:

1. construction/import;
2. first call, including JAX tracing and compilation where applicable; and
3. repeated warm calls.

JAX results are synchronized with `block_until_ready()` before the timer is
stopped. Array sizes are varied because the fastest scalar implementation is
not necessarily the fastest vector implementation.

Run the suite with:

```bash
python -m pip install -e ".[benchmark]"
python -m baldr.benchmarks --dtype float64 --output benchmark.json
```

For a quick smoke run:

```bash
python -m baldr.benchmarks --sizes 1,8 --repeat 2 --number 10
```

The JSON output records Python, platform, package versions, timing parameters,
results, and skipped optional implementations. Benchmark outputs should not be
committed as universal performance claims: hardware and package versions are
part of every result.

## Interpretation rules

- Compare scalar Python-call overhead separately from array throughput.
- Do not include JAX compilation in warm-call timings.
- Compare matching dtypes before drawing conclusions.
- Add CDF and PPF tail-accuracy checks before benchmarking those methods.
- Benchmark a realistic multi-parameter prior transform before selecting the
  default backend.
- Backend selection should occur during distribution construction, not inside
  each hot-path method, unless measurements show the dispatch cost is
  negligible.
