# Optimiser experiments

`auto` (variable projection + LM, with structural exceptions) remains the fitter. The opt-in
`exp_*` policies tried on 2026-11-26 were not promoted. Their code, scripts and full log
(`docs/baseline/rust-experiments.md`) are on the local branch `experiments/v0-v15`.

Figures are suite-average success at 5 ms on the dev gate, against 85.98% for `auto`.

| Tried | Outcome |
|---|---|
| Full-space restart after convergence | +0.3 pp, about 3x the median fit time |
| Routing solvers by equation or starting-point features | static rules lost success; the probe rule gained nothing and was slower |
| Early stop, with or without restart | alone -0.6 pp; with restart +0.7 to +0.8 pp, but worse on NIST |
| Timed 3 ms restart (V5) | passed on dev; on held-out tiers, worse Stringer success and SAGE-Fit ranking |
| Progress or rejection signals to stop or restart (V6-V10) | no clear gain over V5 |
| Combined speed-ups and restart (V15) | 87.39%; gain over V5 below the acceptance margins |

## Worth bringing back

Buffer reuse (V3, `exp_v3_reuse`) and in-place gradient and step updates (V11,
`exp_v11_in_place`) cut time to target and reproduced the same final losses on jointly completed
fits. The one-column scalar shortcut (V12) is faster but changes some results.
