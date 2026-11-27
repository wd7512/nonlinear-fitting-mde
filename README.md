# symbfitrs

A fast CPU fitter for nonlinear equations, written in Rust, for symbolic regression. `uv pip install symbfitrs` (Linux, macOS), then `symbfitrs.compile("a*exp(-b*x) + c").fit(x, y, budget_ms=5)`, or `symbfitrs.Pool(threads=8).fit_many([(expr, x, y), ...])` for batches.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wd7512/symbfitrs/main/docs/benchmark-dark.png"><img alt="Share of fits reaching the target against time per fit for twelve solvers: symbfitrs 72% at 1 ms and 86% at 5 ms; the best other, SciPy lm, 47% and 71%" src="https://raw.githubusercontent.com/wd7512/symbfitrs/main/docs/benchmark-light.png"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wd7512/symbfitrs/main/docs/benchmark-threads-dark.png"><img alt="Share of symbfitrs fits reaching the target against time per fit with 1 to 5 fits running in parallel: 72% at 1 ms and 84% at 5 ms with one at a time, 71% and 82% with five in parallel" src="https://raw.githubusercontent.com/wd7512/symbfitrs/main/docs/benchmark-threads-light.png"></picture>
<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wd7512/symbfitrs/main/docs/benchmark-throughput-dark.png"><img alt="Speed-up in clearing a queue of 5,078 fits with 1 to 5 workers: 1.99x with 2 workers, 2.95x with 3, 3.91x with 4 and 4.20x with 5; wall time 108 s with one worker, 26 s with five" src="https://raw.githubusercontent.com/wd7512/symbfitrs/main/docs/benchmark-throughput-light.png"></picture>

Six suites (NIST StRD, Moré–Garbow–Hillstrom, FitBenchmarking, SAGE-Fit, Stringer 2019, synthetic up to 99 parameters); one thread per fit; every solver at its library defaults (SciPy, Optimistix, Ceres via pyceres, SAGE-Fit, VarProPy, the Rust `varpro` crate; see `bench/baselines.py`); compile time not counted. The second chart reruns symbfitrs through the package on a machine with 4 reported CPUs, on the five suites whose problems are single expressions (not MGH). The third chart queues 5,078 fits on N workers (100 ms budget each); fits that settle on their own give identical results at every N. SAGE-Fit data is Python code the benchmark executes, so fetch only data you trust. The benchmark needs [uv](https://docs.astral.sh/uv/) and [cargo](https://rustup.rs/).

```bash
uv sync && uv run python -m bench.fetch nist fitbenchmarking sagefit stringer2019  # 860 MB
for s in nist_strd mgh fitbenchmarking synthetic_scale sagefit_spb stringer2019_gratings; do
  uv run python -m bench run $s && uv run python -m bench run $s --fitter; done
uv run python -m bench.figure
```

MIT licence; downloaded benchmark data keeps its own.
