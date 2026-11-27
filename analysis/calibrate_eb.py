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
FINE = np.geomspace(P.GRID[0], P.GRID[-1], 61)


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


def profile_cells(cells, Y, starts, ms, sd=None):
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
    np.save('analysis/explore/profile_careful.npy', {'cells': cells, 'L': Lh})


def step4():
    print('--- step 4: heteroscedastic noise: weights 1/sd per bin (the true SE on synthetic data)')
    import time; t0 = time.time(); cells = np.arange(X.N_SUB)
    L = {h: profile_cells(cells, X.DATA['synth'][h], 3, 30.0, sd=X._sd_syn) for h in ('even', 'odd')}
    print(f'  {time.time() - t0:.0f}s')
    np.save('analysis/explore/profile_weighted.npy', L)
    Lf = {h: refine(L[h]) for h in L}
    g1, g2 = npmle2(np.concatenate([Lf['even'], Lf['odd']]))
    report('weighted, shared prior (both halves)', *posterior(Lf['even'], g1, g2, FINE))
    g1, g2 = npmle2(Lf['odd'])
    report('weighted, prior from the odd half', *posterior(Lf['even'], g1, g2, FINE))


def step6():
    print('--- step 6: weighted likelihood + log-normal prior by marginal likelihood (no tuning knob)')
    Lw = np.load('analysis/explore/profile_weighted.npy', allow_pickle=True).item()
    Lf = {h: refine(Lw[h]) for h in Lw}
    g, mu, sd = lognormal_prior(np.concatenate([Lf['even'], Lf['odd']]))
    print(f'  fitted prior: log p ~ N({mu:.2f}, {sd:.2f}^2), median p {np.exp(mu):.2f} (truth: uniform 0.7-3, median 1.85)')
    mean_e, q = posterior(Lf['even'], g, g, FINE)
    report('weighted + log-normal prior (both halves)', mean_e, q)
    g2, *_ = lognormal_prior(Lf['odd'])
    report('weighted + log-normal prior (odd half only)', *posterior(Lf['even'], g2, g2, FINE))
    mean_o, _ = posterior(Lf['odd'], g, g, FINE)
    print(f'  split-half ICC of the posterior mean {X.icc(mean_e, mean_o):.2f}; |diff| > 0.5 in {np.mean(np.abs(mean_e - mean_o) > 0.5):.0%}')


if __name__ == '__main__':
    steps = sys.argv[1:] or ['1', '2']
    Lf = step1() if '1' in steps or '2' in steps else None
    if '2' in steps: step2(Lf)
    if '3' in steps: step3()
    if '4' in steps: step4()
    if '6' in steps: step6()
