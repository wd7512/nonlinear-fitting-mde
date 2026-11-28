"""Calibrating the empirical-Bayes intervals for min(p) (profile_eb.py) on the explore.py subset.

The 90% intervals covered the synthetic truth in only 72% of cells. Each step tests one cause and
prints coverage of the 50/80/90/95% intervals, median 90% width, RMSE and bias of the posterior mean:
  1  grid quantisation: spline the 13x13 profile log-likelihood onto a fine grid, interpolate the CDF
  2  the prior: separate priors for the dominant and weaker peak; prior fitted on the other half
  3  noisy profile: refit the profile on 10 cells with more starts / budget and compare
  4  profiling out the nuisance parameters: Laplace correction -1/2 log det(J'J) at each grid point
  5  self-consistent simulation: truth drawn from the estimated prior

usage: PYTHONPATH=analysis python analysis/calibrate_eb.py [steps, e.g. 1 2]   (default: 1 2)
"""
import sys
import numpy as np
from scipy.interpolate import RectBivariateSpline
import explore as X
import profile_eb as P

LEVELS = (0.5, 0.8, 0.9, 0.95)
LS = np.load('analysis/explore/profile_L.npy', allow_pickle=True).item()
TRUTH_MIN = X.TRUTH[:, 8:10].min(1)
FINE = np.geomspace(P.GRID[0], P.GRID[-1], 81 if X.C.UNCAPPED else 61)
SUF = X.SUF


def refine(L, grid=P.GRID, fine=FINE):
    """Bicubic spline of each cell's log-likelihood in (log p1, log p2), evaluated on `fine`."""
    lg, lf = np.log(grid), np.log(fine)
    out = np.empty((len(L), len(fine), len(fine)))
    for i, Li in enumerate(L):
        s = RectBivariateSpline(lg, lg, Li, kx=3, ky=3)
        assert np.allclose(s(lg, lg), Li, atol=1e-9 * max(1, np.abs(Li).max()))
        out[i] = s(lf, lf)
    return out


def npmle2(L, shared=True, iters=500):
    """Priors g1 (axis 1) and g2 (axis 2); shared=True ties them (profile_eb.npmle)."""
    lik = np.exp(L - L.max((1, 2), keepdims=True))
    k = L.shape[1]; g1 = g2 = np.full(k, 1 / k)
    for _ in range(iters):
        w = lik * g1[None, :, None] * g2[None, None, :]; w /= w.sum((1, 2), keepdims=True)
        m1, m2 = w.sum((0, 2)), w.sum((0, 1))
        if shared:
            g1 = g2 = (m1 + m2) / (m1 + m2).sum()
        else:
            g1, g2 = m1 / m1.sum(), m2 / m2.sum()
    return g1, g2


def lognormal_prior(L, grid=None):
    """Shared prior on the grid: a normal in log p, mean and SD by maximum marginal likelihood
    (grid search). Two parameters, both learned from the population; returns (g, mu, sd)."""
    grid = FINE if grid is None else grid
    lg = np.log(grid); lik = np.exp(L - L.max((1, 2), keepdims=True))
    best = (-np.inf, None)
    for mu in np.linspace(lg[0], lg[-1], 41):
        for sd in np.geomspace(0.05, 2.0, 30):
            g = np.exp(-0.5 * ((lg - mu) / sd) ** 2); g /= g.sum()
            ll = np.log(np.einsum('iab,a,b->i', lik, g, g) + 1e-300).sum()
            if ll > best[0]: best = (ll, (g, mu, sd))
    return best[1]


def posterior(L, g1, g2, grid):
    """Posterior mean and central intervals of min(p1, p2), quantiles from the interpolated CDF."""
    lik = np.exp(L - L.max((1, 2), keepdims=True))
    w = lik * g1[None, :, None] * g2[None, None, :]; w /= w.sum((1, 2), keepdims=True)
    mn = np.minimum(grid[:, None], grid[None, :]).ravel()
    order = np.argsort(mn, kind='stable'); vals = mn[order]
    W = w.reshape(len(w), -1)[:, order]
    mean = W @ vals
    q = {}
    for lev in LEVELS:
        a = (1 - lev) / 2; lo, hi = np.empty(len(W)), np.empty(len(W))
        for i, wi in enumerate(W):
            c = np.cumsum(wi); c = (c - wi / 2)            # mid-point CDF, so a point mass is not one-sided
            lo[i], hi[i] = np.interp(a, c, vals), np.interp(1 - a, c, vals)
        q[lev] = (lo, hi)
    return mean, q


def report(name, mean, q, truth=TRUTH_MIN):
    e = mean - truth
    cov = ' '.join(f'{int(lev * 100)}%:{np.mean((truth >= lo - 1e-9) & (truth <= hi + 1e-9)):4.0%}' for lev, (lo, hi) in q.items())
    print(f'{name:44s} coverage {cov}   90% width {np.median(q[0.9][1] - q[0.9][0]):.2f}   '
          f'RMSE {np.sqrt(np.mean(e ** 2)):.2f}  bias {np.median(e):+.2f}')


def dominant_first(L, Y):
    """Transpose cells so axis 1 is the peak the estimator calls stronger."""
    A = X.EST(Y)[:, 2:4]
    return np.where((A[:, 1] > A[:, 0])[:, None, None], np.swapaxes(L, 1, 2), L)


def step1():
    print('--- step 1: grid quantisation')
    L = {h: LS[f'synth_{h}'] for h in ('even', 'odd')}
    g1, g2 = npmle2(np.concatenate([L['even'], L['odd']]))
    report('coarse 13x13, snapped (as profile_eb.py)', *posterior(L['even'], g1, g2, P.GRID))
    Lf = {h: refine(L[h]) for h in L}
    g1, g2 = npmle2(np.concatenate([Lf['even'], Lf['odd']]))
    report('spline to 61x61, interpolated CDF', *posterior(Lf['even'], g1, g2, FINE))
    return Lf


def step2(Lf):
    print('--- step 2: the prior')
    D = {h: dominant_first(Lf[h], X.DATA['synth'][h]) for h in Lf}
    g1, g2 = npmle2(np.concatenate([D['even'], D['odd']]), shared=False)
    report('separate priors, dominant / weaker peak', *posterior(D['even'], g1, g2, FINE))
    g1, g2 = npmle2(Lf['odd'])
    report('shared prior fitted on the odd half only', *posterior(Lf['even'], g1, g2, FINE))


def profile_cells(cells, Y, starts, ms, sd=None, keep=None):
    """Profile log-likelihood (len(cells), g, g) as in profile_eb.profile, with chosen effort."""
    import symbfitrs
    V = P.V; rng = np.random.default_rng(0); P0 = X.to_variant(X.EST(Y), V)
    pool = symbfitrs.Pool(threads=4); out = np.empty((len(cells), len(P.GRID), len(P.GRID)))
    for a, pa in enumerate(P.GRID):
        for b, pb in enumerate(P.GRID):
            jobs = []
            for i in cells:
                S = X.starts(P0[i], rng, dict(V, starts=starts)); S[:, 8], S[:, 9] = pa, pb
                jobs.append((V['model'], X.TH[i], Y[i], S[:, V['order']]) + (() if sd is None else (1 / sd[i],)))
            res = pool.fit_many(jobs, budget_ms=ms * starts, bounds=dict(V['bounds'], p1=(pa, pa), p2=(pb, pb)), progress=False)
            # weighted: loss is the mean of (r / sd)^2 per bin, so the log-likelihood is -n loss / 2
            out[:, a, b] = [-X.NB * r.loss / (2 * X.SIGMA2['synth'][i]) if sd is None else -X.NB * r.loss / 2
                            for i, r in zip(cells, res)]
            if keep is not None:
                keep[:, a, b] = [r.params for r in res]          # in the model's parameter order
    return out


def step3():
    print('--- step 3: is the profile noisy? 10 cells, 3 starts x 30 ms (cached) vs 8 starts x 300 ms')
    cells = np.arange(0, X.N_SUB, 6)[:10]
    Lc = LS['synth_even'][cells]
    import time; t0 = time.time()
    Lh = profile_cells(cells, X.DATA['synth']['even'], 8, 300.0)
    d = Lh - Lc                                   # >= 0 where the cheap profile missed the optimum
    print(f'  {time.time() - t0:.0f}s; log-lik gain from more effort: median {np.median(d):.3f}, 95th pct {np.percentile(d, 95):.2f}, '
          f'max {d.max():.2f}; grid points improved by > 0.5: {np.mean(d > 0.5):.1%}')
    # how sharp is each likelihood? drop from the best to the next grid neighbours
    for name, L in (('cheap', Lc), ('careful', Lh)):
        top = L.reshape(len(L), -1).max(1)
        print(f'  {name:8s}: median log-lik range over the grid {np.median(top - L.reshape(len(L), -1).min(1)):.1f}')
    np.save(f'analysis/explore/profile_careful{SUF}.npy', {'cells': cells, 'L': Lh})


def step4():
    print('--- step 4: heteroscedastic noise: weights 1/sd per bin (the true SE on synthetic data)')
    import time; t0 = time.time(); cells = np.arange(X.N_SUB)
    L = {h: profile_cells(cells, X.DATA['synth'][h], 3, 30.0, sd=X._sd_syn) for h in ('even', 'odd')}
    print(f'  {time.time() - t0:.0f}s')
    np.save(f'analysis/explore/profile_weighted{SUF}.npy', L)
    Lf = {h: refine(L[h]) for h in L}
    g1, g2 = npmle2(np.concatenate([Lf['even'], Lf['odd']]))
    report('weighted, shared prior (both halves)', *posterior(Lf['even'], g1, g2, FINE))
    g1, g2 = npmle2(Lf['odd'])
    report('weighted, prior from the odd half', *posterior(Lf['even'], g1, g2, FINE))


def step6():
    print('--- step 6: weighted likelihood + log-normal prior by marginal likelihood (no tuning knob)')
    Lw = np.load(f'analysis/explore/profile_weighted{SUF}.npy', allow_pickle=True).item()
    Lf = {h: refine(Lw[h]) for h in Lw}
    g, mu, sd = lognormal_prior(np.concatenate([Lf['even'], Lf['odd']]))
    print(f'  fitted prior: log p ~ N({mu:.2f}, {sd:.2f}^2), median p {np.exp(mu):.2f} (truth: uniform 0.7-3, median 1.85)')
    mean_e, q = posterior(Lf['even'], g, g, FINE)
    report('weighted + log-normal prior (both halves)', mean_e, q)
    g2, *_ = lognormal_prior(Lf['odd'])
    report('weighted + log-normal prior (odd half only)', *posterior(Lf['even'], g2, g2, FINE))
    mean_o, _ = posterior(Lf['odd'], g, g, FINE)
    print(f'  split-half ICC of the posterior mean {X.icc(mean_e, mean_o):.2f}; |diff| > 0.5 in {np.mean(np.abs(mean_e - mean_o) > 0.5):.0%}')


def laplace(params, sd, eps=1e-6):
    """-1/2 log det(J'J) of the nuisance parameters (all but p1, p2) at each profile point, J the
    weighted Jacobian by central differences of Model.predict. params: (cells, g, g, n_params)."""
    m = P.V['model']; names = m.params
    nuis = [k for k, n in enumerate(names) if n not in ('p1', 'p2')]
    out = np.empty(params.shape[:3])
    for i in range(params.shape[0]):
        x = X.TH[i]
        for a in range(params.shape[1]):
            for b in range(params.shape[2]):
                th = params[i, a, b]; J = np.empty((len(x), len(nuis)))
                for c, k in enumerate(nuis):
                    h = eps * max(1.0, abs(th[k])); tp, tm = th.copy(), th.copy(); tp[k] += h; tm[k] -= h
                    J[:, c] = (m.predict(x, tp) - m.predict(x, tm)) / (2 * h) / sd[i]
                sign, logdet = np.linalg.slogdet(J.T @ J + 1e-10 * np.eye(len(nuis)))
                out[i, a, b] = -0.5 * logdet
    return out


def step7():
    print('--- step 7: Laplace correction for the profiled-out nuisance parameters (weighted, log-normal prior)')
    import time; t0 = time.time(); cells = np.arange(X.N_SUB); n = len(P.V['model'].params)
    L, C = {}, {}
    for h in ('even', 'odd'):
        par = np.empty((X.N_SUB, len(P.GRID), len(P.GRID), n))
        L[h] = profile_cells(cells, X.DATA['synth'][h], 3, 30.0, sd=X._sd_syn, keep=par)
        C[h] = laplace(par, X._sd_syn)
    print(f'  {time.time() - t0:.0f}s; correction range per cell (median): {np.median(np.ptp(C["even"].reshape(X.N_SUB, -1), 1)):.1f} log units')
    np.save(f'analysis/explore/profile_laplace{SUF}.npy', {'L': L, 'C': C})
    for name, Lx in (('weighted profile', L), ('weighted profile + Laplace', {h: L[h] + C[h] for h in L})):
        Lf = {h: refine(Lx[h]) for h in Lx}
        g, mu, sd = lognormal_prior(np.concatenate([Lf['even'], Lf['odd']]))
        me, q = posterior(Lf['even'], g, g, FINE); mo, _ = posterior(Lf['odd'], g, g, FINE)
        report(f'{name}', me, q)
        print(f'    prior median p {np.exp(mu):.2f}; split-half ICC {X.icc(me, mo):.2f}')


def continuation(Y, sd, par, L0, ms=30.0):
    """Second pass: refit every grid point from the optima of its 8 neighbours (p set to this
    point's values); keep the better of the two passes. par: (cells, g, g, n) model order."""
    import symbfitrs
    V = P.V; names = V['model'].params; k1, k2 = names.index('p1'), names.index('p2')
    pool = symbfitrs.Pool(threads=4); g = len(P.GRID); L = L0.copy(); par = par.copy()
    for a, pa in enumerate(P.GRID):
        for b, pb in enumerate(P.GRID):
            nb = [(a + da, b + db) for da in (-1, 0, 1) for db in (-1, 0, 1) if (da or db) and 0 <= a + da < g and 0 <= b + db < g]
            jobs = []
            for i in range(X.N_SUB):
                S = np.array([par[i, u, v] for u, v in nb]); S[:, k1], S[:, k2] = pa, pb
                jobs.append((V['model'], X.TH[i], Y[i], S, 1 / sd[i]))
            res = pool.fit_many(jobs, budget_ms=ms * len(nb), bounds=dict(V['bounds'], p1=(pa, pa), p2=(pb, pb)), progress=False)
            for i, r in enumerate(res):
                if -X.NB * r.loss / 2 > L[i, a, b]:
                    L[i, a, b] = -X.NB * r.loss / 2; par[i, a, b] = r.params
    return L, par


def step8():
    print('--- step 8: profile continuation (refit each grid point from its neighbours) (weighted, log-normal prior)')
    import time; t0 = time.time(); n = len(P.V['model'].params); L, L2 = {}, {}
    for h in ('even', 'odd'):
        par = np.empty((X.N_SUB, len(P.GRID), len(P.GRID), n))
        L[h] = profile_cells(np.arange(X.N_SUB), X.DATA['synth'][h], 3, 30.0, sd=X._sd_syn, keep=par)
        L2[h], _ = continuation(X.DATA['synth'][h], X._sd_syn, par, L[h])
        d = L2[h] - L[h]
        print(f'  {h}: points improved by > 0.5 log units {np.mean(d > 0.5):.1%}, max {d.max():.1f}', flush=True)
    print(f'  {time.time() - t0:.0f}s')
    np.save(f'analysis/explore/profile_continued{SUF}.npy', L2)
    for name, Lx in (('one pass', L), ('with continuation', L2)):
        Lf = {h: refine(Lx[h]) for h in Lx}
        g, mu, sd = lognormal_prior(np.concatenate([Lf['even'], Lf['odd']]))
        me, q = posterior(Lf['even'], g, g, FINE); mo, _ = posterior(Lf['odd'], g, g, FINE)
        report(name, me, q); print(f'    prior median p {np.exp(mu):.2f}; split-half ICC {X.icc(me, mo):.2f}')


import consistency as C


def real_sd(rows, half):
    """Per-bin SE of the half's mean curve from a per-cell mean-variance line var = c0 + c1 mean
    fitted to the trial variances (4 trials per bin are too few for a raw per-bin SD)."""
    k = 0 if half == 'even' else 1
    out = []
    for i in rows:
        t = C.CELLS[i][1][k]; mu, var = t.mean(1), t.var(1, ddof=1)
        A = np.stack([np.ones_like(mu), mu], 1); c, *_ = np.linalg.lstsq(A, var, rcond=None)
        v = np.clip(A @ c, 0.1 * var.mean(), None)
        out.append(np.sqrt(v / t.shape[1]))
    return np.array(out)


def profile_rows(rows, half, sd, starts=3, ms=30.0):
    import symbfitrs
    V = P.V; rng = np.random.default_rng(0); Y = C.HALF[half][rows]
    P0 = X.to_variant(np.array([C.estimator(C.TH[i], C.HALF[half][i]) for i in rows]), V)
    pool = symbfitrs.Pool(threads=4); out = np.empty((len(rows), len(P.GRID), len(P.GRID)))
    for a, pa in enumerate(P.GRID):
        for b, pb in enumerate(P.GRID):
            jobs = []
            for j, i in enumerate(rows):
                S = X.starts(P0[j], rng, dict(V, starts=starts)); S[:, 8], S[:, 9] = pa, pb
                jobs.append((V['model'], C.TH[i], Y[j], S[:, V['order']], 1 / sd[j]))
            res = pool.fit_many(jobs, budget_ms=ms * starts, bounds=dict(V['bounds'], p1=(pa, pa), p2=(pb, pb)), progress=False)
            out[:, a, b] = [-C.TH.shape[1] * r.loss / 2 for r in res]
    return out


def step9(rows=None, tag='sub60'):
    rows = X.SUB if rows is None else rows
    print(f'--- step 9: real data, {len(rows)} neurons: weighted profile (mean-variance noise model) + log-normal prior')
    import time; t0 = time.time(); L = {}
    for h in ('even', 'odd'):
        sd = real_sd(rows, h); L[h] = profile_rows(rows, h, sd)
        # noise-model check: best-fit weighted MSE should be about 1 if the variances are right
        print(f'  {h}: median best weighted MSE {np.median(-2 * L[h].reshape(len(rows), -1).max(1) / C.TH.shape[1]):.2f} (1 = noise model right)', flush=True)
    print(f'  {time.time() - t0:.0f}s')
    np.save(f'analysis/explore/profile_real_{tag}{SUF}.npy', {'rows': rows, **L})
    # the mean-variance line overestimates the noise (best weighted MSE ~0.62 where ~0.95 is
    # expected), so rescale each cell's noise to its own residuals (a no-op when it is right)
    n = C.TH.shape[1]
    L = {h: L[h] / (-2 * L[h].reshape(len(rows), -1).max(1) / n * n / (n - 10))[:, None, None] for h in L}
    Lf = {h: refine(L[h]) for h in L}
    g, mu, sdp = lognormal_prior(np.concatenate([Lf['even'], Lf['odd']]))
    me, q = posterior(Lf['even'], g, g, FINE); mo, qo = posterior(Lf['odd'], g, g, FINE)
    w95 = q[0.95][1] - q[0.95][0]
    print(f'  population: log p ~ N({mu:.2f}, {sdp:.2f}^2) -> median p {np.exp(mu):.2f}, '
          f'central 80% of neurons {np.exp(mu - 1.28 * sdp):.2f}-{np.exp(mu + 1.28 * sdp):.2f}')
    print(f'  per cell: median 95% width {np.median(w95):.2f}; narrower than 1.0 in {np.mean(w95 < 1):.0%}; '
          f'split-half ICC {X.icc(me, mo):.2f}; |diff| > 0.5 in {np.mean(np.abs(me - mo) > 0.5):.0%}')
    # do the halves' 95% intervals overlap (a consistency check that needs no truth)?
    lo_e, hi_e = q[0.95]; lo_o, hi_o = qo[0.95]
    print(f'  even and odd 95% intervals overlap in {np.mean((lo_e <= hi_o) & (lo_o <= hi_e)):.0%} of neurons')
    # the notebook's population: double-peaked cells (estimator amplitude ratio >= 0.1 in both halves)
    A = {h: np.array([C.estimator(C.TH[i], C.HALF[h][i])[2:4] for i in rows]).clip(min=0) for h in ('even', 'odd')}
    dbl = np.all([A[h].min(1) / np.maximum(A[h].max(1), 1e-12) >= 0.1 for h in A], 0)
    pm = (me + mo) / 2
    boot = [np.median(pm[dbl][np.random.default_rng(b).integers(0, dbl.sum(), dbl.sum())]) for b in range(2000)]
    print(f'  double-peaked cells: {dbl.sum()} of {len(rows)}; median posterior min p {np.median(pm[dbl]):.2f} '
          f'(95% CI {np.percentile(boot, 2.5):.2f}-{np.percentile(boot, 97.5):.2f})')
    return dict(mu=mu, sd=sdp, me=me, mo=mo)


if __name__ == '__main__':
    steps = sys.argv[1:] or ['1', '2']
    Lf = step1() if '1' in steps or '2' in steps else None
    if '2' in steps: step2(Lf)
    if '3' in steps: step3()
    if '4' in steps: step4()
    if '6' in steps: step6()
    if '7' in steps: step7()
    if '8' in steps: step8()
    if '9' in steps: step9()
    if '300' in steps: step9(np.arange(C.N), 'all300')
