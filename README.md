# symbfitrs

A fast CPU fitter for nonlinear equations, written in Rust, for symbolic regression: you propose
the equations, it fits their parameters in milliseconds.

**Not installable yet**: the Python package is in progress ([roadmap](docs/roadmap.md)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmark-dark.png">
  <img alt="Share of fits reaching the target against time per fit: symbfitrs 74% at 1 ms and 87% at 5 ms; SciPy trf 45% and 68%, SciPy lm 48% and 73%" src="docs/benchmark-light.png">
</picture>

Six suites: NIST StRD, Moré–Garbow–Hillstrom, FitBenchmarking, SAGE-Fit, Stringer et al. (2019)
and synthetic equations with up to 99 parameters. Each fit runs on one thread; SciPy's
`least_squares` uses its defaults. Equation compile time is not counted for either side.

SAGE-Fit candidate equations are Python code that the benchmark executes: only fetch data you trust.

## Reproduce

Needs [uv](https://docs.astral.sh/uv/) and [cargo](https://rustup.rs/).

```bash
uv sync
uv run python -m bench.fetch nist fitbenchmarking sagefit stringer2019   # stringer2019 is 860 MB
for s in nist_strd mgh fitbenchmarking synthetic_scale sagefit_spb stringer2019_gratings; do
  uv run python -m bench run $s && uv run python -m bench run $s --fitter
done
uv run python -m bench.figure
```

## License

MIT. Downloaded benchmark data keeps its own licence.
