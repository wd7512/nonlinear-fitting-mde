# Data

Drop datasets here (e.g. `data/<name>/curves.csv`) and describe them with a TOML
config in `configs/`. **Everything in `data/` except this file and `examples/` is
git-ignored** — remove the `data/*` lines from `.gitignore` to version your data.

## Supported files

| Format | Extensions | Notes |
|---|---|---|
| CSV / TSV | `.csv`, `.tsv` | `read_options` go to `pandas.read_csv` (e.g. `{ comment = "#" }`) |
| Whitespace text | `.txt`, `.dat` | NIST-style columns (`format = "whitespace"`) |
| Parquet / Feather | `.parquet`, `.feather` | needs `uv sync --extra parquet` |
| JSON / JSON lines | `.json`, `.jsonl` | records orientation |
| NumPy | `.npz` | each array becomes a column |

Rows with missing values in the used columns are dropped per curve.

## Layouts

**long** — one row per observation. Optional curve id and per-point sigma:

```
curve,x,y,sigma
c00,0.0,1.93,0.02
c00,0.07,1.88,0.02
c01,0.0,0.84,0.02
```

**wide** — shared x column(s), one y column per curve (`layout = "wide"`,
`y_columns` defaults to every non-x column).

Multiple input variables: `x = ["u", "v"]`, then use `u` and `v` in expressions.

## Config reference (`configs/*.toml`)

```toml
name = "my_experiment"          # results go to results/<name>/
seed = 42
solvers = ["adam", "lbfgs", "gauss_newton", "lm", "trf", "varpro_lm"]
budgets_ms = [1, 5, 10, 25]     # checkpoints; the last is the per-run deadline
targets = [1e-6, 1e-10]         # success: loss - reference <= target (normalised MSE)
worker_deadline_s = 7.5         # stop launching new fits (cooperative)
watchdog_s = 9.0                # hard kill of the worker

[[datasets]]
kind = "table"
name = "my_curves"
path = "data/my_curves/curves.csv"   # relative to the repo root (or the config file)
format = "auto"                      # or csv / tsv / whitespace / parquet / feather / json / jsonl / npz
layout = "long"                      # or "wide"
x = "x"                              # or ["u", "v"]
y = "y"
curve = "curve"                      # optional: one problem per curve
sigma = "sigma"                      # optional: weighted residuals (y - f) / sigma
max_curves = 1000                    # optional subsample
curve_sample = "first"               # or "random" (seeded)
reference = "auto"                   # auto | given | best_found | none (see below)

[[datasets.models]]                  # one or more candidate equations per dataset
name = "damped_cos"
expr = "A * exp(-k * x) * cos(w * x) + c"   # SymPy syntax; exp, log, sin, tanh, sqrt, ...
params = ["A", "k", "w", "c"]
# variables = ["x"]                  # default: the dataset's x columns
# linear = []                        # override auto-detected VarPro linear params
starts.values = { guess = [1, 0.5, 2, 0] }  # explicit named starts, and/or:
starts.bounds = { A = [0.5, 2.5], k = [0.05, 1.5], w = [1, 6], c = [-0.5, 0.5] }
starts.n_random = 2                  # uniform draws from bounds (also sets Adam/L-BFGS scaling)
# starts.near_scale = 0.05           # "near" start around reference.params
# reference.params = [...]  |  reference.sse = 0.1245  |  reference.loss = 1e-3

# Built-in families instead of an expression:
# [[datasets.models]]
# family = "exp_mixture"             # exp_mixture | sin_mixture | coupled_tanh
# p = 8
```

### Reference minimum

| `reference` | Meaning |
|---|---|
| `given` | from `reference.params` / `.sse` (unweighted RSS, e.g. NIST certified) / `.loss` |
| `best_found` | best loss reached by any solver/start on that problem (post hoc; "matched the best") |
| `none` | raw normalised loss vs. target (right for noise-free data) |
| `auto` | `given` if the model has a reference, else `best_found` |

Check a new config before running it:

```bash
uv run symreg-rs inspect configs/my_experiment.toml
```

It validates the file, prints each model (parameters, detected linear parameters),
curve counts, start losses, the reference and the worst-case run time. If it warns
that the run may be cut short, raise `worker_deadline_s`/`watchdog_s` or use `max_curves`.

## Built-in benchmark datasets

`scripts/fetch_benchmarks.py` downloads these into `data/external/` (git-ignored). Each is a
dataset `kind` usable in any config (see `configs/benchmarks/`):

| kind | keys |
|---|---|
| `nist` | `path` (dir of .dat files), `problems` ("all" or names), `reference` ("certified" / "best_found") |
| `mgh` | `problems` ("all", names or numbers), `scales` (starts = scale × x0; default [1, 10, 100]) |
| `sagefit_spb` | `path` (SAGE-Fit checkout), `split` (train/test/ood_test), `max_problems`, `random_starts` |
| `fitbenchmarking` | `path` (…/examples/benchmark_problems), `sets` (sub-folders), `use_errors` |
| `synthetic` | `families`, `param_counts` (any p ≥ 2; odd p adds a constant offset), `n_curves` |
| `stringer2019_gratings` | `path` (gratings.npz), `max_cells`, `cell_sample`, `equations`, `n_perturbed` |

Top-level options for these: `success` = "absolute" | "relative" | "reduction",
`relative_floor`, and per-dataset `loss_norm` = "mean_square" | "variance" | "sum_of_squares" | "mse".
