# symbfitrs

A fast CPU fitter for nonlinear equations, written in Rust, for use inside symbolic regression:
you propose the equations, it fits their parameters on millisecond budgets (variable projection
+ Levenberg–Marquardt, up to about 100 parameters).

**Status: not installable yet.** The solver works and is benchmarked, but the Rust equation
compiler and the Python package are still to be built. See [`docs/roadmap.md`](docs/roadmap.md).

## Benchmark

Needs [uv](https://docs.astral.sh/uv/) and [cargo](https://rustup.rs/).

```bash
uv sync
uv run python -m bench run quick            # SciPy baselines on synthetic equations
uv run python -m bench run quick --fitter   # the Rust fitter on the same problems
```

Results go to `results/<name>/` (or `<name>_rust/`): `summary.md`, `records.jsonl` (every fit
with its loss history) and `success_vs_time.png`.

| command | what it does |
|---|---|
| `python -m bench run CONFIG [--fitter]` | fit every problem, then report |
| `python -m bench report CONFIG [--fitter]` | rebuild the report from saved records |
| `python -m bench gate build` / `gate run` | fixed dev/test cases for changes to the fitter |

`run --jobs` shares the fits between 4 workers. It is faster, but timings get noisier.

### Suites

```bash
uv run python -m bench.fetch                # downloads into data/external/
uv run python -m bench run nist_strd
```

| config | data | success means |
|---|---|---|
| `nist_strd` | [NIST StRD](https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml), 27 problems × 2 starts | within 1e-6 of the certified minimum |
| `mgh` | Moré–Garbow–Hillstrom (1981), 35 problems × 3 starts (built in) | Moré–Wild reduction test |
| `synthetic_scale` | exponential, sine and tanh families, up to 99 parameters (built in) | loss below 1e-6 (noise-free) |
| `sagefit_spb` | [SAGE-Fit](https://github.com/CAS-CLab/SAGE-Fit), 20 curves × 10 candidate equations | within SAGE-Fit's "lost" threshold; ranking accuracy |
| `fitbenchmarking` | [FitBenchmarking](https://github.com/fitbenchmarking/fitbenchmarking) Mantid problems | within 1e-3 of the best found |
| `stringer2019_gratings` | [Stringer et al. 2019](https://doi.org/10.25378/janelia.8279387) V1 neurons × 10 tuning equations | within 1% of the best MSE found |

`bench.fetch` takes suite names (`nist`, `fitbenchmarking`, `sagefit`, `stringer2019`); without
arguments it fetches the first three. The Stringer recording is 860 MB. SAGE-Fit candidates are
Python code that gets executed, so only run data you trust.

### Baselines

One registered function in [`bench/baselines.py`](bench/baselines.py):

```python
@baseline
def scipy_trf(job):
    return least_squares(job.residual, job.p0, method="trf").x
```

### How it measures

- **One thread per fit.** Fits run in a shuffled order after a warm-up.
- **No late credit.** A checkpoint at B ms counts only losses available by B ms.
- **Loss** is the mean squared weighted residual, normalised per dataset.
- **SciPy runs at its defaults** (finite-difference Jacobian, default tolerances).
- **Compile time is not yet counted** for either side. The end-to-end, string-to-parameters
  comparison comes with the Python package (roadmap step 4), so do not read the current numbers
  as an end-to-end speed claim.

### Gate

To change the fitter, run a tier before and after, then compare:

```bash
uv run python -m bench gate run --tier dev
uv run python -m bench gate run --tier dev --baseline results/gate/<earlier run>.jsonl
```

The acceptance margins are fixed in advance in [`bench/gate.py`](bench/gate.py).

## Development

```bash
uv run pytest
uv run ruff check && uv run ruff format --check
cargo test --release --manifest-path rust/Cargo.toml
```

## License

MIT. Downloaded benchmark data keeps its own licence.
