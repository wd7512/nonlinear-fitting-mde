# Versioned Rust `auto` speed experiments

`auto` is unchanged. `auto_v1` names the same VarPro+LM policy; later
versions are opt-in and cumulative. The default benchmark policy list and
the behavior of the other Rust solvers stay the same.

| Policy | Change from the previous version |
|---|---|
| `auto_v1` | Exact alias for `auto`. |
| `auto_v1.1` | Take ownership of the existing full-rank projection basis instead of copying it. |
| `auto_v1.2` | Reuse the parameter buffer when building the projected Jacobian. |
| `auto_v1.3` | Scale Jacobian columns in their existing workspace. |
| `auto_v1.4` | Reuse the gradient vector across iterations. |
| `auto_v1.5` | Write damped trial parameters directly into the trial buffer. |

These changes do not introduce solver selection, change tolerances, or
change how rank-deficient projection is calculated. Timed restarts and the
one-column direct solve are not included: the former showed a held-out
tradeoff, and the latter changed some final losses.

Run the stack on a locally built [gate](gate.md):

```sh
uv run nlfit-mde gate run --tier dev \
  --policies auto,auto_v1,auto_v1.1,auto_v1.2,auto_v1.3,auto_v1.4,auto_v1.5
```

For repeated timings with an explicit order seed, build the Rust binary in
release mode and pass `--seed` to `nlfit-rs bench`. Neither third-party gate
inputs nor raw benchmark data are tracked.

## Dev results

Apple Silicon, one thread, frozen 552-case dev gate, shuffle seeds 42–44.
Each run completed 3,864 fits. Success gives each benchmark suite equal
weight. The time ratio is the geometric mean of the per-case ratios on
fits both the candidate and `auto_v1` solved by 5 ms.

| Policy | 5 ms success | Paired time vs `auto_v1` | Median fit (ms) |
|---|---:|---:|---:|
| `auto` | 85.84–85.98% | 1.005–1.006× | 2.239–2.350 |
| `auto_v1` | 85.98% | 1.000× | 2.257–2.325 |
| `auto_v1.1` | 85.86–85.98% | 0.952–0.961× | 2.139–2.251 |
| `auto_v1.2` | 85.98% | 0.957–0.962× | 2.083–2.151 |
| `auto_v1.3` | 85.98–86.11% | 0.943–0.951× | 2.001–2.117 |
| `auto_v1.4` | 85.98–86.11% | 0.946–0.955× | 2.036–2.156 |
| `auto_v1.5` | 85.98–86.11% | 0.946–0.964× | 2.087–2.113 |

Every revision matched v1's solver status, final parameters, recorded loss
sequence, and evaluation counts on every jointly completed fit (517–518
per run). Wall-clock checkpoint results varied slightly, even between the
two baseline aliases. `auto_v1.3` was the strongest measured speed variant:
4.9–5.7% less time to target. V1.4 and v1.5 are separately measurable but
did not improve on v1.3 here. No default switches or new held-out runs.
