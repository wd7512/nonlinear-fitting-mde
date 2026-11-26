# nlfit-mde

How fast can a CPU fit a nonlinear equation? `nlfit-mde` races least-squares optimisers on
millisecond budgets (1, 5, 10, 25, 100 ms per fit) across standard benchmark suites and your
own data, and reports how often each one reaches the target in time.

Optimisers: Adam, L-BFGS, Gauss–Newton, Levenberg–Marquardt (LM), trust-region reflective
(TRF), variable projection + LM (VarPro), and a single-threaded Rust optimiser (`rust/`) that
defaults to VarPro + LM.

## Quick start

Needs [uv](https://docs.astral.sh/uv/); the Rust optimiser also needs
[cargo](https://rustup.rs/).

```bash
uv sync
uv run nlfit-mde run mde_synthetic          # 144 fits on synthetic equations, a few seconds
uv run nlfit-mde run mde_synthetic --rust   # the same problems with the Rust optimiser
```

Results go to `results/<name>/summary.md` (tables), `records.jsonl` (every fit with its loss
history) and `success_vs_time.png`.

## Commands

| command | what it does |
|---|---|
| `nlfit-mde run CONFIG [--rust]` | fit every problem with every optimiser, then report |
| `nlfit-mde report CONFIG [--rust]` | rebuild the report from saved records |
| `nlfit-mde inspect CONFIG` | validate a config and preview its problems |
| `nlfit-mde gate build` / `gate run` | fixed dev/test cases for changes to the Rust optimiser |
| `nlfit-mde gate compare --challenger lm` | where another Rust policy beats `auto`, by problem feature |

`run --jobs` shares the fits between 4 single-threaded workers. It is much faster, but the
timings get noisier, so use one worker when comparing speeds.

`CONFIG` is a path or a name from `configs/`. With `--rust`, results go to
`results/<name>_rust/` and are merged with the Python results of the same config if they exist.

## Benchmark suites

```bash
uv run python scripts/fetch_benchmarks.py        # downloads into data/external/
uv run nlfit-mde run nist_strd
```

| config | data | success means |
|---|---|---|
| `nist_strd` | [NIST StRD](https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml), 27 problems × 2 starts | within 1e-6 of the certified minimum |
| `mgh` | Moré–Garbow–Hillstrom (1981), 35 problems × 3 starts (built in) | Moré–Wild reduction test |
| `synthetic_scale` | exponential, sine and tanh families, up to 99 parameters (built in) | loss below 1e-6 (noise-free) |
| `sagefit_spb` | [SAGE-Fit](https://github.com/CAS-CLab/SAGE-Fit), 20 curves × 10 candidate equations | within SAGE-Fit's "lost" threshold; ranking accuracy |
| `fitbenchmarking` | [FitBenchmarking](https://github.com/fitbenchmarking/fitbenchmarking) Mantid problems | within 1e-3 of the best found |
| `stringer2019_gratings` | [Stringer et al. 2019](https://doi.org/10.25378/janelia.8279387) V1 neurons × 10 tuning equations | within 1% of the best MSE found (true equation unknown) |

`fetch_benchmarks.py` takes suite names (`nist`, `fitbenchmarking`, `sagefit`,
`stringer2019`); without arguments it fetches the first three. The Stringer recording is
860 MB.

## Your own data

Put a table in `data/`, write a config in `configs/`, check it with `inspect`, then `run` it.
CSV, TSV, Parquet, JSON and NPZ files work, with many curves per file, optional per-point
errors and several candidate equations written as expressions.
See [`data/README.md`](data/README.md) and `configs/examples/`.

Expressions are parsed with SymPy and Python candidate functions are executed, so only run
configs and data you trust.

## How it measures

- **One thread, one fit at a time.** Fits run in a shuffled order after a warm-up.
- **No late credit.** Each loss is timestamped; a checkpoint at B ms counts only losses
  available by B ms.
- **Loss** is the mean squared weighted residual, normalised per dataset. Success compares it
  with a reference: a certified value, zero for noise-free data, or the best any optimiser found.
- **Fixed defaults.** No optimiser is tuned on the benchmarks.
- **The Rust optimiser** runs on equations exported from SymPy as compiled instruction lists.
  Problems without a symbolic form are skipped there.

The original experiment design is in [`docs/MDE.md`](docs/MDE.md).

## Development

```bash
uv run pytest
uv run ruff check && uv run ruff format --check
cargo test --release --manifest-path rust/Cargo.toml
```

To change what the Rust optimiser does by default, compare on the gate (`--tier quick`, `dev`,
`test`, `rank`). The acceptance margins are fixed in advance in
`src/nlfit_mde/rust_bridge/gate.py`.

For opt-in speed comparisons, see the [versioned `auto` stack](docs/baseline/auto-v1.md).
`auto` itself stays on the original policy.

## License

MIT. Downloaded benchmark data keeps its own licence.
