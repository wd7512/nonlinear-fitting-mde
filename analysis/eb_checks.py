"""Checks on the empirical-Bayes exponents, from the saved profiles (no new fits).

1  robust split-half agreement (Spearman, ICC on log p, share within 0.5) and the cells that
   drive the uncapped synthetic ICC of -0.05
2  sensitivity to the prior family: log-normal (the method), gamma, a two-component log-normal
   mixture, and the nonparametric prior with a fixed smoothing (a check only)
3  empirical Bayes as arm G of the 300-neuron consistency table, capped and uncapped

usage: PYTHONPATH=analysis python analysis/eb_checks.py
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
import calibrate_eb as K
import explore as X
import consistency as C

GRIDS = {'capped': np.geomspace(0.4, 4.0, 13), 'uncapped': np.geomspace(0.4, 16.0, 17)}
SUF = {'capped': '', 'uncapped': '_pinf'}
TRUTH = X.TRUTH[:, 8:10].min(1)
NB = C.TH.shape[1]
rng0 = np.random.default_rng(0)


def fine_of(grid):
    return np.geomspace(grid[0], grid[-1], 61 if grid[-1] <= 4 else 81)


def marg_ll(lik, g):
    return np.log(np.einsum('iab,a,b->i', lik, g, g) + 1e-300).sum()


def prior_lognormal(lik, fine):
    lg = np.log(fine); best = (-np.inf, None)
    for mu in np.linspace(lg[0], lg[-1], 41):
        for s in np.geomspace(0.05, 2.0, 30):
            g = np.exp(-0.5 * ((lg - mu) / s) ** 2); g /= g.sum()
            ll = marg_ll(lik, g)
            if ll > best[0]: best = (ll, g)
    return best[1]


def prior_gamma(lik, fine):
    """Gamma density on p, as weights on the log grid (pdf x p): shape k, mean m."""
    best = (-np.inf, None)
    for m in np.geomspace(0.5, 8.0, 30):
        for k in np.geomspace(0.5, 60.0, 30):
            lg = k * np.log(fine) - k * fine / m; g = np.exp(lg - lg.max()); g /= g.sum()
            ll = marg_ll(lik, g)
            if ll > best[0]: best = (ll, g)
    return best[1]


def prior_mixture(lik, fine):
    """Two log-normal components, weights and both (mu, sd) by maximum marginal likelihood."""
    lg = np.log(fine)
    def g_of(t):
        m1, s1, m2, s2, a = t[0], np.exp(t[1]), t[2], np.exp(t[3]), 1 / (1 + np.exp(-t[4]))
        g = a * np.exp(-0.5 * ((lg - m1) / s1) ** 2) / s1 + (1 - a) * np.exp(-0.5 * ((lg - m2) / s2) ** 2) / s2
        return g / g.sum()
    best = None
    for m1, m2 in ((0.2, 1.0), (0.0, 0.8), (0.5, 1.5)):
        r = minimize(lambda t: -marg_ll(lik, g_of(t)), [m1, np.log(0.3), m2, np.log(0.3), 0.0], method='Nelder-Mead',
                     options=dict(maxiter=2000, xatol=1e-4, fatol=1e-6))
        if best is None or r.fun < best.fun: best = r
    return g_of(best.x)


def prior_npmle_smoothed(L, fine, bw=0.15):
    g, _ = K.npmle2(L)
    lg = np.log(fine); kern = np.exp(-0.5 * ((lg[:, None] - lg[None, :]) / bw) ** 2)
    g = kern @ g; return g / g.sum()


PRIORS = {'log-normal (method)': prior_lognormal, 'gamma': prior_gamma, '2-component log-normal': prior_mixture,
          'nonparametric, smoothed (check)': None}


def lik_of(L):
    return np.exp(L - L.max((1, 2), keepdims=True))


def agreement(a, b):
    la, lb = np.log(a), np.log(b)
    return dict(icc=X.icc(a, b), icc_log=X.icc(la, lb), spearman=spearmanr(a, b)[0], within05=np.mean(np.abs(a - b) <= 0.5))


def rescale(L):
    best = -2 * L.reshape(len(L), -1).max(1) / NB
    return L / (best * NB / (NB - 10))[:, None, None]


def load(kind, data):
    if data == 'synth':
        L = np.load(f'analysis/explore/profile_weighted{SUF[kind]}.npy', allow_pickle=True).item()
    else:
        d = np.load(f'analysis/explore/profile_real_all300{SUF[kind]}.npy', allow_pickle=True).item()
        L = {h: rescale(d[h]) for h in ('even', 'odd')}
    return {h: K.refine(L[h], GRIDS[kind], fine_of(GRIDS[kind])) for h in ('even', 'odd')}


def double_cells():
    A = {h: np.array([C.estimator(C.TH[i], C.HALF[h][i])[2:4] for i in range(C.N)]).clip(min=0) for h in ('even', 'odd')}
    return np.all([A[h].min(1) / np.maximum(A[h].max(1), 1e-12) >= 0.1 for h in A], 0)


def boot_ci(x, n=2000):
    r = np.random.default_rng(1)
    v = [np.median(x[r.integers(0, len(x), len(x))]) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])


if __name__ == '__main__':
    out = {}
    DBL = double_cells()
    for kind in ('capped', 'uncapped'):
        fine = fine_of(GRIDS[kind])
        S = load(kind, 'synth'); R = load(kind, 'real')
        print(f'\n===== {kind} (p grid {GRIDS[kind][0]:.1f}-{GRIDS[kind][-1]:.0f})')
        for name, fn in PRIORS.items():
            res = {}
            for tag, D in (('synth', S), ('real', R)):
                both = np.concatenate([D['even'], D['odd']]); lik = lik_of(both)
                g = prior_npmle_smoothed(both, fine) if fn is None else fn(lik, fine)
                me, q = K.posterior(D['even'], g, g, fine); mo, qo = K.posterior(D['odd'], g, g, fine)
                res[tag] = dict(me=me, mo=mo, q=q, qo=qo, g=g, ll=marg_ll(lik, g))
            s, r = res['synth'], res['real']
            e = s['me'] - TRUTH
            cov = {lev: np.mean((TRUTH >= lo - 1e-9) & (TRUTH <= hi + 1e-9)) for lev, (lo, hi) in s['q'].items()}
            ag_s, ag_r = agreement(s['me'], s['mo']), agreement(r['me'], r['mo'])
            pm = (r['me'] + r['mo']) / 2; ci = boot_ci(pm[DBL])
            print(f'-- prior: {name}')
            print(f'   synthetic: RMSE {np.sqrt(np.mean(e ** 2)):.2f}, bias {np.median(e):+.2f}, coverage 90% {cov[0.9]:.0%} 95% {cov[0.95]:.0%}; '
                  f'split-half ICC {ag_s["icc"]:.2f}, ICC(log p) {ag_s["icc_log"]:.2f}, Spearman {ag_s["spearman"]:.2f}, within 0.5 {ag_s["within05"]:.0%}')
            print(f'   real 300 (G): split-half ICC {ag_r["icc"]:.2f}, ICC(log p) {ag_r["icc_log"]:.2f}, Spearman {ag_r["spearman"]:.2f}, '
                  f'within 0.5 {ag_r["within05"]:.0%}; double-peaked ({DBL.sum()}) median min p {np.median(pm[DBL]):.2f} '
                  f'(95% CI {ci[0]:.2f}-{ci[1]:.2f}); marginal log-lik {r["ll"]:.1f}')
            out[(kind, name)] = dict(synth_rmse=float(np.sqrt(np.mean(e ** 2))), synth_bias=float(np.median(e)),
                                     synth_cov90=float(cov[0.9]), synth_cov95=float(cov[0.95]), synth_agree=ag_s,
                                     real_agree=ag_r, real_median=float(np.median(pm[DBL])), real_ci=ci.tolist(), real_ll=float(r['ll']))
            if name.startswith('log-normal') and kind == 'uncapped':
                # which cells drive the synthetic ICC: largest |even - odd| posterior means
                d = np.abs(s['me'] - s['mo']); top = np.argsort(d)[::-1][:5]
                print('   largest synthetic split-half gaps (cell: even, odd, truth):',
                      ', '.join(f'{i}: {s["me"][i]:.2f}, {s["mo"][i]:.2f}, {TRUTH[i]:.2f}' for i in top))
                keep = np.ones(len(d), bool); keep[top[:3]] = False
                print(f'   ICC without those 3 cells: {X.icc(s["me"][keep], s["mo"][keep]):.2f}')
    np.save('analysis/explore/eb_checks.npy', {f'{k}|{n}': v for (k, n), v in out.items()}, allow_pickle=True)
    print('\nsaved analysis/explore/eb_checks.npy')
