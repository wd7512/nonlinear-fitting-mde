# Roadmap

`symbfitrs` is a fast Rust fitter for nonlinear equations that you can install, plus a benchmark
in this repository that shows it is fast and behaves correctly. The symbolic regression search
that proposes equations is the user's job.

## Interface

```python
model = symbfitrs.compile("a*exp(-b*x) + c", variables=["x"])
result = model.fit(x, y, budget_ms=5)

pool = symbfitrs.Pool(threads=8)
results = pool.fit_many([(expr, x, y), ...], budget_ms=5)  # same order as the jobs
```

- Parsing, differentiation and linear-parameter detection happen in Rust.
- `fit_many` runs one job per thread, compiles a repeated equation once, shows a `tqdm` bar
  (`progress=False` to silence it) and reports a failed job in its result without stopping the batch.
- `budget_ms` starts when a thread picks the job up.
- Required dependencies: NumPy and `tqdm`.

## v0.1 is ready when

1. It installs from PyPI on Linux and macOS without Rust.
2. `compile`, `fit` and `Pool.fit_many` work.
3. Gate success matches [`baseline/gate.md`](baseline/gate.md) within the margins in `bench/gate.py`.
4. The benchmark calls the installed package, compares string-to-parameters time against SciPy,
   and reports throughput and success at 1, 4 and 8 threads.

Later: streaming results, Windows, buffer-reuse speed-ups, parameter bounds, user-defined
functions.

## Layout

```
rust/              the fitter: parser, differentiation, solver, thread pool, bindings
python/symbfitrs/  the installed package
bench/             suites, gate, baselines, reports; never shipped
tests/
```

## Baselines

A baseline is one registered function in `bench/baselines.py`:

```python
@baseline
def scipy_trf(job):
    return least_squares(job.residual, job.p0, method="trf").x
```

## Order of work

| step | work | done when |
|---|---|---|
| 1 | split out `bench/`, baseline registry | done |
| 2 | Rust parser, differentiation, linear detection | done (`rust/src/expr.rs`) |
| 3 | Python bindings, thread pool, progress bar | done (`rust/src/python.rs`, `python/symbfitrs`) |
| 4 | benchmark calls the package; end-to-end and thread-count comparisons | all six suites run |
| 5 | wheel builds and PyPI release | wheels build in CI; a `v*` tag publishes |

The SymPy tape export and the `symbfitrs-bench` program stay until step 4 replaces them.
