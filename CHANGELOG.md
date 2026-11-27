# Changelog

Changes to the installed `symbfitrs` package. The benchmark in `bench/` is not shipped and is
not listed here.

## Unreleased

## 0.1.1 (2026-11-27)

- `FitResult.history_ms` and `history_loss`: the best loss at the first evaluation and at each
  improvement, timed from the call like `time_ms`.
- `FitResult.n_nonfinite`: evaluations that gave a non-finite loss.
- `Model.predict(x, params)`.
- `Pool.fit_many` jobs can be `(model, x, y, p0, weights)`, and `model` can be a compiled
  `Model` with its own variables.
- Expressions: `atan2`, `wrap`, `step` and `expwerfc(a, b)`. A declared parameter named `pi`
  wins over the constant.
- Faster VarPro+LM steps: three redundant copies are skipped, with the same solver trajectories.

## 0.1.0 (2026-11-26)

First release: `compile`, `Model.fit` and `Pool.fit_many`, with wheels for Linux and macOS
(x86_64, aarch64).
