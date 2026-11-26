# Minimum decisive experiment: CPU nonlinear equation fitting

## Question

Which optimiser most reliably fits smooth nonlinear equations with fewer than 100 free parameters, under a small CPU-time budget? The eventual use case is scoring candidate equations across 1,000 independent curves in class symbolic regression.

**Target:** one benchmark invocation in under 10 seconds on a single CPU core (aim for 3–5 seconds once dependencies are loaded). No GPU, compilation, training, or expensive reference optimisation in the timed run.

## 1. Algorithms

| ID | Optimiser | Reason |
|---|---|---|
| A | Adam (full batch) | Cheap first-order iterations |
| B | L-BFGS | Cheap approximate curvature |
| C | Gauss–Newton | Exploits least-squares structure |
| D | Levenberg–Marquardt (LM) | Damped least-squares baseline |
| E | Trust-region reflective (TRF) | Robust least-squares alternative |
| F | Variable projection + LM (VarPro) | Eliminates conditionally linear parameters |

Use unconstrained FP64 fitting initially. VarPro falls back to LM when there are no conditionally linear parameters; label these results accordingly. Use fixed, documented solver defaults (e.g. Adam learning rate 0.01 on scaled parameters); do not tune on the test cases. Multi-start performance can be computed from the same two runs without running them again.

## 2. Tiny deterministic benchmark

Generate 12 noise-free problems: 3 equation families × 4 parameter counts (2, 8, 32, 96). Use two shared initialisations per problem (one near and one far), fixed seed 42, and `m = max(64, 4*p)` observations. Total: 144 solver runs.

| Family | Equation | Purpose |
|---|---|---|
| Exponential mixture | `sum(a[k] * exp(-b[k] * x))` | Separable, correlated parameters |
| Sinusoidal mixture | `sum(a[k] * sin(b[k] * x))` | Separable, multiple local minima |
| Coupled nonlinear | `tanh(sum(theta[k] * phi[k](x)))` | Nonseparable; larger dense Jacobians |

For the mixtures use `p/2` terms; for the coupled model use `p` fixed, scaled basis functions (e.g. cosine basis on a uniform grid). Generate bounded true parameters and finite starts; scale inputs, outputs and parameter units consistently. Preserve the same parameter domain for all methods (no solver-specific clipping); a non-finite trial must be handled by the solver or recorded as a failure. Include moderate frequencies/rates and avoid overflow in fixture generation.

Generate `y = f(x, theta_true)` in FP64. Hence the reference minimum is zero up to rounding. Compare prediction loss, not parameter error, because different parameter vectors may represent the same curve. High-order mixtures can be deliberately poorly identifiable; report these separately rather than assuming every parameter is recoverable.

## 3. Runtime budget

- Single thread for all solvers and BLAS; deterministic run order, shuffled with the fixed seed.
- Maximum 25 ms of optimisation per run; record checkpoints at 1, 5, 10 and 25 ms using a monotonic high-resolution wall clock.
- Nominal solver budget: `144 * 25 ms = 3.6 s`. Allow a few seconds for setup, instrumentation and iteration overshoot.
- Compile/prepare expression evaluators and Jacobians once per problem; warm up all solvers once on a separate tiny fixture. Report setup separately and also report total invocation time.
- Use a cooperative deadline check at each iteration/evaluation, and a global watchdog (e.g. 9 seconds). A native call may overrun its deadline; log actual time, do not credit late results at earlier checkpoints. If the global watchdog truncates the run, report incomplete coverage rather than counting missing cases as failures.
- Do not launch a fresh subprocess per fit. For a strict wall-clock cap, run the whole benchmark in one worker under a parent watchdog; stream completed records so partial results survive termination.

## 4. Metrics

Use normalised mean squared error:

```
loss(theta) = mean((f(x, theta) - y)^2) / max(mean(y^2), 1e-12)
```

Primary: proportion of runs reaching `loss <= 1e-6` within each time budget. Also report a stricter target (`1e-10`). Count non-finite results and timeouts explicitly. For the multi-start view, success is the best of the two starts, with their CPU budgets added (sequential execution).

Secondary:

- Median time to target (successful fits only, alongside success rate).
- Median and p90 time per completed iteration (solver-defined; not directly equivalent across solvers).
- Function/Jacobian evaluations and their cumulative time; report these separately because iterations do different amounts of work.
- Best loss at each checkpoint, by equation family and parameter count.
- End-to-end fits/s (completed solver attempts / elapsed time, including per-fit setup).

Keep every result, including failed runs. Do not compare only converged fits. For the future symbolic-regression application, extend the benchmark with incorrect equations and known/reference minima, then measure pairwise equation ranking errors under matched budgets. This first noise-free benchmark measures fitting reliability, not scientific equation verification or global optimality on arbitrary data.

## 5. Pseudocode

```python
SEED = 42
BUDGETS_MS = [1, 5, 10, 25]
TARGETS = [1e-6, 1e-10]
SOLVERS = [Adam, LBFGS, GaussNewton, LM, TRF, VarProLM]
set_threads(1)  # configure BLAS before importing numerical libraries
rng = RNG(SEED)

# Setup: generate data, analytic/AD Jacobians, and shared initialisations.
problems = []
for family in [ExpMixture, SinMixture, CoupledNonlinear]:
    for p in [2, 8, 32, 96]:
        x = linspace(0, 1, max(64, 4*p))
        model = prepare(family, p, x)
        theta_true = sample_true_parameters(model, rng)
        y = model.predict(theta_true)
        starts = make_near_and_far_starts(theta_true, rng)
        problems.append(Problem(model, y, starts))

warm_up(SOLVERS, separate_fixture)
jobs = shuffled([(p, s, a) for p in problems
                 for s in p.starts for a in SOLVERS], rng)

for problem, theta0, algorithm in jobs:
    if global_deadline_reached():
        break  # report incomplete benchmark

    t0 = clock_ns()
    history = [(0, loss(problem, theta0))]  # actual initial loss/time tracked
    counters = Counters()
    # All callbacks share the same prediction and Jacobian kernels.
    # Record every finite trial loss when it becomes available, including
    # rejected trials; retain the best parameters actually evaluated.
    evaluator = TimedEvaluator(problem, history, counters, t0, deadline_ms=25)
    try:
        solver = initialise(algorithm, theta0.copy(), evaluator)
        while not solver.finished and elapsed_ms(t0) < 25:
            with counters.time_iteration():
                solver.step()  # checks deadline in evaluation callbacks too
    except DeadlineExceeded:
        pass
    except NumericalFailure:
        counters.failure = True

    # Never use a loss evaluated after a checkpoint to score that checkpoint.
    best = best_so_far_at(history, BUDGETS_MS)
    save_record(problem.id, algorithm, theta0.id, best, history,
                counters, elapsed_ms(t0))

summarise_success_rates(TARGETS, BUDGETS_MS)
summarise_iteration_times_and_evaluations()
summarise_time_to_target_and_batch_throughput()
plot_success_rate_vs_time()  # optional; outside timed solver loop
```

Implementation note: the pseudocode describes an instrumentable solver interface, not existing Python APIs. SciPy's solvers can be wrapped with residual/Jacobian callbacks and a controlled deadline exception; exact iteration timing is not always exposed. For VarPro, reconstruct full parameters and score the original residual, including the cost of the inner linear solve; use QR/SVD rather than an explicit matrix inverse.

## 6. Decision rule

Prefer the method with the highest success rate at 5–25 ms, and use iteration time and throughput as tie-breakers. Inspect per-family results rather than hiding failures in a pooled average. If a cheap solver performs well at 1–5 ms but a slower solver dominates at 25 ms, test a cheap-to-robust fallback in the next experiment (e.g. Adam or GN → LM).

The benchmark is a smoke test, not a statistical ranking. Only after a promising result, add more seeds, noisy and misspecified models, NIST cases, and batched 1,000-curve throughput. Prototype using NumPy/SciPy (and a small shared analytic/AD evaluator); port the winning strategy to Rust if warranted.
