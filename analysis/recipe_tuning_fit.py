"""Drop-in CPU fitter for the jacob_tuning notebooks, built on symbfitrs bounds and multi-start.

Paste into a notebook after the models / estimators cells (it needs `MODELS`, `P_MIN`, `P_MAX`,
`MIN_WIDTH`, `MAX_WIDTH`, `DEVICE` from there) and replace `fit`:

    fit = recipe_fit                       # same signature and return value as the notebooks' fit
    params['Gen. Gaussian'] = fit('Gen. Gaussian', angles, R_train)
    p_lo, p_hi = bootstrap_p(angles, R_train, params['Gen. Gaussian'])   # 95% intervals of min(p)
    eb = eb_exponents(angles, R_train, params['Gen. Gaussian'])          # empirical Bayes (preferred)
    eb['min_p'], eb['lo95'], eb['hi95'], eb['population']              # per cell, and the population

Set `P_CAP = np.inf` before fitting to remove the exponents' upper limit (the notebooks use 4).

What it does differently from both notebooks:
* the model has no clips: the same limits are bounds on the parameters (`bounds=`), so no
  parameter can get stuck in a zero-gradient clip region;
* every cell is fitted from 8 starts (the estimator, the estimator with both exponents set to
  0.7, 1.4, 2 and 3, and 3 random restarts) and the lowest loss is kept;
* it fits to convergence: no early stopping, no penalty, nothing tuned on test data.
Everything runs on the CPU, one cell per core.
"""
import numpy as np
import torch
import symbfitrs

N_STARTS, BUDGET_MS, SEED = 8, 300.0, 0
P_CAP = None          # None: the notebook's P_MAX; np.inf: no upper limit on p


def _peak(offset, left, right, power):
    d = f'(2*atan(tan((x - (theta + {offset}))/2)))'          # circular distance to the peak
    width = left if left == right else f'({right} + ({left} - {right})*max(-sign({d}), 0))'
    return f'exp(-0.5*(max(abs({d}), 1e-12)/{width})^{power})'


def _model(name):
    """(compiled model without clips, notebook parameter order, bounds by name)"""
    if name == 'Gaussian':
        names = ('theta', 'b', 'a1', 'a2', 'w')
        expr = f'b + a1*{_peak(0, "w", "w", 2)} + a2*{_peak("pi", "w", "w", 2)}'
        bounds = {'b': (0, None), 'a1': (0, None), 'a2': (0, None), 'w': (0.01, None)}
    else:
        names = ('theta', 'b', 'a1', 'a2', 'l1', 'r1', 'l2', 'r2', 'p1', 'p2')
        expr = f'b + a1*{_peak(0, "l1", "r1", "p1")} + a2*{_peak("pi", "l2", "r2", "p2")}'
        bounds = {'b': (0, None), 'a1': (0, None), 'a2': (0, None),
                  **{k: (MIN_WIDTH, MAX_WIDTH) for k in ('l1', 'r1', 'l2', 'r2')},  # noqa: F821
                  'p1': (P_MIN, _p_hi()), 'p2': (P_MIN, _p_hi())}                     # noqa: F821
    return symbfitrs.compile(expr), names, bounds


def _p_hi():
    hi = P_MAX if P_CAP is None else P_CAP                                      # noqa: F821
    return None if not np.isfinite(hi) else hi


def _starts(p0, names, rng):
    p0 = np.asarray(p0, np.float64).copy()
    p0[0] %= 2 * np.pi
    S = [p0]
    if 'p1' in names:
        for p in (0.7, 1.4, 2.0, 3.0):
            q = p0.copy(); q[8:10] = p; S.append(q)
    while len(S) < N_STARTS:
        q = p0.copy()
        q[0] = (q[0] + rng.normal(0, 0.15)) % (2 * np.pi)
        q[1:4] *= np.exp(rng.normal(0, 0.3, 3))
        q[4:] *= np.exp(rng.normal(0, 0.6, len(q) - 4))
        S.append(q)
    return np.array(S)


def _fit_curves(model_name, angles, Y, starts_from=None, seed=SEED):
    """Fit each row of Y; returns (n_cells, n_params) in the notebook's order and the results."""
    _, estimator = MODELS[model_name]                                   # noqa: F821
    model, names, bounds = _model(model_name)
    order = [names.index(p) for p in model.params]; inv = np.argsort(order)
    rng = np.random.default_rng(seed)
    jobs, cells = [], []
    for i, y in enumerate(Y):
        ok = ~np.isnan(y)
        if not ok.any():
            continue
        p0 = estimator(angles[ok], y[ok]) if starts_from is None else starts_from[i]
        jobs.append((model, angles[ok], y[ok], _starts(p0, names, rng)[:, order]))
        cells.append(i)
    res = symbfitrs.Pool().fit_many(jobs, budget_ms=BUDGET_MS * N_STARTS, bounds=bounds, progress=False)
    P = np.full((len(Y), len(names)), np.nan)
    for i, r in zip(cells, res, strict=True):
        if r.params.size:
            P[i] = r.params[inv]
    return P, res


def recipe_fit(model, angles, Y):
    """Same signature and return type as the notebooks' `fit`."""
    P, res = _fit_curves(model, angles, Y)
    st = {}
    for r in res: st[r.status] = st.get(r.status, 0) + 1
    print(f'  bounded multi-start fit: {st}')
    return torch.as_tensor(P, dtype=torch.float32, device=DEVICE)  # noqa: F821


def bootstrap_p(angles, Y, P, n_boot=100, seed=1):
    """95% interval of min(p1, p2) per cell by a wild residual bootstrap: each bootstrap curve is
    the fit plus the residuals with random signs, refitted from the original fit's parameters.
    Works with a single repeat per half (no trials needed). Peaks are paired to the original fit."""
    P = np.asarray(P.cpu() if hasattr(P, 'cpu') else P, np.float64)
    f, _ = MODELS['Gen. Gaussian']                                      # noqa: F821
    with torch.no_grad():
        fitted = f(torch.as_tensor(angles, dtype=torch.float64), torch.as_tensor(P)).numpy()
    resid = Y - fitted
    rng = np.random.default_rng(seed)
    mins = np.empty((n_boot, len(Y)))
    for b in range(n_boot):
        Yb = fitted + resid * rng.choice([-1.0, 1.0], resid.shape)
        Pb, _ = _fit_curves('Gen. Gaussian', angles, Yb, starts_from=P, seed=seed + b)
        swap = np.abs(np.arctan2(np.sin(P[:, 0] - Pb[:, 0]), np.cos(P[:, 0] - Pb[:, 0]))) > np.pi / 2
        pb = np.clip(Pb[:, 8:10], P_MIN, P_MAX)                         # noqa: F821
        pb = np.where(swap[:, None], pb[:, ::-1], pb)
        mins[b] = pb.min(1)
    return np.nanpercentile(mins, 2.5, 0), np.nanpercentile(mins, 97.5, 0)


# ---------------------------------------------------------------- empirical Bayes for the exponents
def eb_exponents(angles, Y, P, grid=None, starts=3, ms=30.0):
    """Empirical-Bayes posterior of min(p1, p2) per cell, and the population distribution of p.

    1. Noise per bin: each cell's squared residuals from its fit `P` (notebook order, e.g. from
       `recipe_fit`) regressed on the fitted curve, var = c0 + c1 * f (works with one repeat).
    2. Profile: (p1, p2) pinned on `grid` x `grid` (bounds lo = hi), everything else refitted
       with residuals weighted by 1 / noise SD; rescaled so the best point's weighted MSE is the
       expected (n - 10) / n.
    3. Prior: log p ~ Normal(mu, sd), mu and sd by maximum marginal likelihood over all cells
       (both peaks share it; nothing hand-set). Posterior per cell on a fine grid.
    Returns min_p (posterior mean), lo95 / hi95 (2.5% / 97.5%; covered the truth in about 88% of
    synthetic cells, so read them as roughly 90% intervals) and population (mu, sd, median p)."""
    from scipy.interpolate import RectBivariateSpline
    P = np.asarray(P.cpu() if hasattr(P, 'cpu') else P, np.float64)
    hi = _p_hi()
    grid = np.geomspace(0.4, 4.0 if hi is not None and hi <= 4 else 16.0, 13 if hi is not None and hi <= 4 else 17) \
        if grid is None else np.asarray(grid, float)
    model, names, bounds = _model('Gen. Gaussian')
    order = [names.index(p) for p in model.params]
    f, _ = MODELS['Gen. Gaussian']                                      # noqa: F821
    with torch.no_grad():
        fit = f(torch.as_tensor(angles, dtype=torch.float64), torch.as_tensor(P)).numpy()
    ok = ~np.isnan(Y)
    sd = np.empty_like(fit)
    for i in range(len(Y)):
        r2, fi = (Y[i] - fit[i])[ok[i]] ** 2, fit[i][ok[i]]
        c, *_ = np.linalg.lstsq(np.stack([np.ones_like(fi), fi], 1), r2, rcond=None)
        sd[i] = np.sqrt(np.clip(c[0] + c[1] * fit[i], 0.1 * r2.mean(), None))
    rng = np.random.default_rng(SEED)
    pool = symbfitrs.Pool()
    n_bins = ok.sum(1)
    L = np.full((len(Y), len(grid), len(grid)), -np.inf)
    for a, pa in enumerate(grid):
        for b, pb in enumerate(grid):
            jobs = []
            for i in range(len(Y)):
                S = _starts(P[i], names, rng)[:starts].copy(); S[:, 8], S[:, 9] = pa, pb
                jobs.append((model, angles[ok[i]], Y[i][ok[i]], S[:, order], 1 / sd[i][ok[i]]))
            res = pool.fit_many(jobs, budget_ms=ms * starts, bounds=dict(bounds, p1=(pa, pa), p2=(pb, pb)), progress=False)
            L[:, a, b] = [-n_bins[i] * r.loss / 2 if r.params.size else -np.inf for i, r in enumerate(res)]
    best_mse = -2 * L.reshape(len(L), -1).max(1) / n_bins
    L = L / (best_mse * n_bins / (n_bins - 10))[:, None, None]
    fine = np.geomspace(grid[0], grid[-1], 81)
    lg, lf = np.log(grid), np.log(fine)
    Lf = np.stack([RectBivariateSpline(lg, lg, Li)(lf, lf) for Li in L])
    lik = np.exp(Lf - Lf.max((1, 2), keepdims=True))
    best = (-np.inf, None)
    for mu in np.linspace(lf[0], lf[-1], 41):
        for s in np.geomspace(0.05, 2.0, 30):
            g = np.exp(-0.5 * ((lf - mu) / s) ** 2); g /= g.sum()
            ll = np.log(np.einsum('iab,a,b->i', lik, g, g) + 1e-300).sum()
            if ll > best[0]: best = (ll, (g, mu, s))
    g, mu, s = best[1]
    w = lik * g[None, :, None] * g[None, None, :]; w /= w.sum((1, 2), keepdims=True)
    mn = np.minimum(fine[:, None], fine[None, :]).ravel(); o = np.argsort(mn, kind='stable'); vals = mn[o]
    W = w.reshape(len(w), -1)[:, o]
    cdf = np.cumsum(W, 1) - W / 2
    lo = np.array([np.interp(0.025, c, vals) for c in cdf]); hi95 = np.array([np.interp(0.975, c, vals) for c in cdf])
    return dict(min_p=W @ vals, lo95=lo, hi95=hi95, population=dict(mu=mu, sd=s, median_p=float(np.exp(mu))))
