"""Profile likelihood + empirical Bayes for the exponents, on the explore.py subset.

1. Profile: for each cell, fix (p1, p2) on a grid (bounds with lo = hi) and fit everything else
   with bounds + starts; RSS(p1, p2) -> log-likelihood -RSS / (2 sigma^2), sigma^2 = the cell's
   noise variance of the mean curve from its trials.
2. Empirical Bayes: one population distribution g(p) on the grid, shared by both peaks, fitted by
   EM to all cells' likelihoods (the nonparametric MLE). No hand-set strength: how much each cell
   leans on the population follows from how flat its own likelihood is.
3. Per cell: posterior mean and 90% interval of min(p1, p2); compared with the estimator and
   plain least squares on synthetic data (known truth) and for split-half agreement on real data.

usage: PYTHONPATH=analysis python analysis/profile_eb.py
"""
import time
import numpy as np
import symbfitrs
import explore as X

GRID = np.geomspace(0.4, 16.0, 17) if X.C.UNCAPPED else np.geomspace(0.4, 4.0, 13)   # P_CAP=inf: p up to 16
V = X.variant()                      # bounds + starts, sigma widths, p in [0.1, 4]
STARTS = 3


def profile(Y, sigma2):
    """loglik (cells, g, g) over (p1, p2) on GRID, and the profile fits' wall time."""
    allb = np.arange(X.NB); rng = np.random.default_rng(0)
    P0 = X.to_variant(X.EST(Y, allb), V)
    jobs, keys = [], []
    for i in range(X.N_SUB):
        for a, pa in enumerate(GRID):
            for b, pb in enumerate(GRID):
                S = X.starts(P0[i], rng, dict(V, starts=STARTS)); S[:, 8], S[:, 9] = pa, pb
                jobs.append((V['model'], X.TH[i], Y[i], S[:, V['order']], None))
                keys.append((i, a, b))
    t0 = time.time()
    # one bounds dict per job is not supported, so pin p1, p2 per grid point by grouping jobs
    L = np.full((X.N_SUB, len(GRID), len(GRID)), np.nan)
    pool = symbfitrs.Pool(threads=4)
    for a, pa in enumerate(GRID):
        for b, pb in enumerate(GRID):
            idx = [k for k, (i, aa, bb) in enumerate(keys) if aa == a and bb == b]
            bnd = dict(V['bounds'], p1=(pa, pa), p2=(pb, pb))
            res = pool.fit_many([jobs[k][:4] for k in idx], budget_ms=30.0 * STARTS, bounds=bnd, progress=False)
            for k, r in zip(idx, res):
                L[keys[k][0], a, b] = -X.NB * r.loss / (2 * sigma2[keys[k][0]])
    return L, time.time() - t0


def npmle(L, iters=500):
    """Population g on GRID (shared by both peaks) maximising the marginal likelihood."""
    lik = np.exp(L - L.max((1, 2), keepdims=True))
    g = np.full(len(GRID), 1 / len(GRID))
    for _ in range(iters):
        w = lik * g[None, :, None] * g[None, None, :]
        w /= w.sum((1, 2), keepdims=True)
        g = w.sum((0, 2)) + w.sum((0, 1)); g /= g.sum()
    return g


def posterior_minp(L, g):
    lik = np.exp(L - L.max((1, 2), keepdims=True))
    w = lik * g[None, :, None] * g[None, None, :]; w /= w.sum((1, 2), keepdims=True)
    mn = np.minimum(GRID[:, None], GRID[None, :])
    mean = (w * mn).sum((1, 2))
    order = np.argsort(mn.ravel()); cdf = np.cumsum(w.reshape(len(w), -1)[:, order], 1); vals = mn.ravel()[order]
    lo = vals[(cdf < 0.05).sum(1)]; hi = vals[np.minimum((cdf < 0.95).sum(1), len(vals) - 1)]
    mle = mn.ravel()[L.reshape(len(L), -1).argmax(1)]
    return mean, lo, hi, mle


if __name__ == '__main__':
    out = {}
    for d in ('synth', 'real'):
        for h in ('even', 'odd'):
            L, secs = profile(X.DATA[d][h], X.SIGMA2[d])
            out[d, h] = L
            print(f'profile {d} {h}: {secs:.0f}s for {L.size} grid fits', flush=True)
    np.save(f'analysis/explore/profile_L{X.SUF}.npy', {f'{d}_{h}': v for (d, h), v in out.items()}, allow_pickle=True)
    truth = X.TRUTH[:, 8:10].min(1)
    for d in ('synth', 'real'):
        g = npmle(np.concatenate([out[d, 'even'], out[d, 'odd']]))
        print(f'\n{d}: population g(p) on the grid:', ' '.join(f'{p:.2f}:{v:.2f}' for p, v in zip(GRID, g) if v > 0.01))
        res = {h: posterior_minp(out[d, h], g) for h in ('even', 'odd')}
        mean, lo, hi, mle = res['even']
        est = X.EST(X.DATA[d]['even'])[:, 8:10].clip(0.1, 4).min(1)
        if d == 'synth':
            for name, v in (('estimator', est), ('profile MLE (= least squares)', mle), ('empirical Bayes mean', mean)):
                e = v - truth
                print(f'  {name:30s} RMSE {np.sqrt(np.mean(e ** 2)):.2f}  bias {np.median(e):+.2f}  within 0.5 {np.mean(np.abs(e) <= 0.5):.0%}')
            print(f'  90% interval covers the truth in {np.mean((truth >= lo - 1e-9) & (truth <= hi + 1e-9)):.0%} of cells; '
                  f'median width {np.median(hi - lo):.2f}; population median: truth {np.median(truth):.2f}, '
                  f'EB {np.median(mean):.2f}, estimator {np.median(est):.2f}, LSQ {np.median(mle):.2f}')
        for name, k in (('profile MLE', 3), ('EB mean', 0)):
            a, b = res['even'][k], res['odd'][k]
            print(f'  split-half {name:12s}: ICC {X.icc(a, b):.2f}, |diff| > 0.5 in {np.mean(np.abs(a - b) > 0.5):.0%}')
        print(f'  cells whose 90% interval is narrower than 1.0: {np.mean(hi - lo < 1.0):.0%}')
