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
3. Gate success matches today's `auto` within the margins in `bench/gate.py`.
4. The benchmark calls the installed package, compares string-to-parameters time against SciPy,
   and reports throughput and success at 1, 4 and 8 threads.

Later: streaming results, Windows, buffer-reuse speed-ups (see `experiments.md`), parameter
bounds, user-defined functions.

## Layout

```
rust/              the fitter: parser, differentiation, solver, thread pool, bindings
python/symbfitrs/  the installed package (arrives with the bindings)
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
| 1 | strip to this design, rename, split out `bench/`, baseline registry | tests pass, gate losses unchanged |
| 2 | Rust parser, differentiation, linear detection | Jacobians match SymPy's, gate holds |
| 3 | Python bindings, thread pool, progress bar | a clean install fits an equation |
| 4 | benchmark calls the package; end-to-end and thread-count comparisons | all six suites run |
| 5 | wheel builds and PyPI release | installs on a machine without `cargo` |

The SymPy tape export and the `symbfitrs-bench` program stay until steps 2 and 3 replace them.
