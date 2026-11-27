"""Do real bounds / structural fixes close the Adam-vs-LM gap?  All methods fit the same 300 train curves from the
same estimator start and are scored identically: MSE of the notebook's own clipped model (torch, float64) at the
returned parameters, against the best result any method found for that neuron.
usage: PYTHONPATH=. python analysis/bounds_experiments.py OUT.npz
"""
import sys, time, itertools
import numpy as np
import torch
from scipy.optimize import least_squares
import symbfitrs
from _neuro import *

OUT = sys.argv[1]
d = np.load('analysis/diagnosis.npz', allow_pickle=True)
P_s, P_a, ab = d['P_s'], d['P_a'], d['ab']                     # symbfitrs (clipped, LM) and Adam solutions from before
N = len(ROWS); cells = [trials(r) for r in ROWS]
TH = np.stack([c[0] for c in cells]); TRAIN = np.stack([c[1][0].mean(1) for c in cells]); TEST = np.stack([c[1][1].mean(1) for c in cells])
est = lambda th, y: np.asarray(estimator(th, y), dtype=float)
P0 = np.array([est(TH[i], TRAIN[i]) for i in range(N)])        # notebook order
nb = list(NAMES)                                                # ('theta','b','a1','a2','l1','r1','l2','r2','p1','p2')
TWO_PI = 2 * np.pi
BOX = {'theta': (-np.inf, np.inf), 'b': (0, np.inf), 'a1': (0, np.inf), 'a2': (0, np.inf), **{k: (0.1, TWO_PI) for k in ('l1', 'r1', 'l2', 'r2')}, 'p1': (0.1, 4.0), 'p2': (0.1, 4.0)}
th64, y64, te64 = (torch.tensor(a, dtype=torch.float64) for a in (TH, TRAIN, TEST))
gg = lambda th, P: gen_gauss(th, P).squeeze(0)


def score(P):                                                   # unified score: the notebook's clipped model, float64
    with torch.no_grad():
        pr = gg(th64, torch.tensor(P, dtype=torch.float64))
        return ((pr - y64) ** 2).mean(1).numpy(), (1 - ((te64 - pr) ** 2).sum(1) / ((te64 - te64.mean(1, keepdim=True)) ** 2).sum(1)).numpy()


# ---- the unclipped model: every clip removed; b, a1, a2 are then linear parameters
e = EXPR; W = '6.283185307179586'
e = e.replace(f'min(max(theta, 0), {W})', 'theta')
for n in ('l1', 'r1', 'l2', 'r2'): e = e.replace(f'min(max({n}, 0.1), {W})', n)
for n in ('p1', 'p2'): e = e.replace(f'min(max({n}, 0.1), 4.0)', n)
i1, i2 = e.index(' + max(a1, 0)*'), e.index(' + max(a2, 0)*')
PEAK1, PEAK2 = e[i1 + len(' + max(a1, 0)*'):i2], e[i2 + len(' + max(a2, 0)*'):]
FREE = symbfitrs.compile(f'b + a1*{PEAK1} + a2*{PEAK2}')
print('unclipped model: linear =', FREE.linear, '| nonlinear =', [p for p in FREE.params if p not in FREE.linear], flush=True)
to_nb = lambda x, m: np.array([x[m.params.index(n)] if n in m.params else 0.0 for n in nb])          # model order -> notebook order
from_nb = lambda v, m: np.array([v[nb.index(n)] for n in m.params])
lo_hi = lambda m: (np.array([BOX[n][0] for n in m.params]), np.array([BOX[n][1] for n in m.params]))
feasible = lambda v: all(BOX[n][0] - 1e-6 <= v[j] <= BOX[n][1] + 1e-6 for j, n in enumerate(nb) if n not in ('a1', 'a2') or v[nb.index(n)] != 0 or True)

res = {}   # method -> dict(P (N,10 notebook order), secs (N,), ok, extra)
def run_scipy(label, model, method, P_start_nb, bounded):
    P = np.zeros((N, 10)); secs = np.zeros(N); ok = np.zeros(N, bool); nfev = np.zeros(N)
    lo, hi = lo_hi(model)
    for i in range(N):
        x0 = from_nb(P_start_nb[i], model)
        if bounded: x0 = np.clip(x0, lo, hi)
        f = lambda x: model.predict(TH[i], x) - TRAIN[i]
        t = time.perf_counter()
        try:
            r = least_squares(f, x0, method=method, bounds=(lo, hi) if bounded else (-np.inf, np.inf)); x = r.x; ok[i] = True; nfev[i] = r.nfev
        except Exception:
            x = x0
        secs[i] = time.perf_counter() - t; P[i] = to_nb(x, model)
    res[label] = dict(P=P, secs=secs, ok=ok, nfev=nfev); print(f'{label:34s} done, median {1e3 * np.median(secs):.1f} ms/fit, failures {int((~ok).sum())}', flush=True)

# X1: SciPy on the clipped model, no bounds (the benchmark's setup); start = raw estimator, as for symbfitrs and Adam
for m in ('lm', 'trf', 'dogbox'):
    run_scipy(f'SciPy {m}, clipped model, no bounds', MODEL, m, P0, False)
# X2: SciPy with true bounds on the unclipped model
P0c = np.clip(P0, [BOX[n][0] for n in nb], [BOX[n][1] for n in nb]); P0c[:, 0] = np.mod(P0c[:, 0], TWO_PI)
for m in ('trf', 'dogbox'):
    run_scipy(f'SciPy {m}, true bounds', FREE, m, P0c, True)

# X3: symbfitrs variable projection on the unclipped model, unconstrained
pool = symbfitrs.Pool()
def run_pool(model, starts_nb, budget=1000.0):
    return pool.fit_many([(model, TH[i], TRAIN[i], from_nb(starts_nb[i], model)) for i in range(N)], budget_ms=budget, progress=False)
t = time.time(); rs = run_pool(FREE, P0c); wall = time.time() - t
Pfree = np.array([to_nb(r.params, FREE) for r in rs]); stat = np.array([r.status for r in rs])
neg = (Pfree[:, 1:4] < -1e-9).any(1); out_w = ((Pfree[:, 4:8] < 0.1 - 1e-6) | (Pfree[:, 4:8] > TWO_PI + 1e-6)).any(1); out_p = ((Pfree[:, 8:10] < 0.1 - 1e-6) | (Pfree[:, 8:10] > 4 + 1e-6)).any(1)
res['symbfitrs VarPro, unclipped, unconstrained'] = dict(P=Pfree, secs=np.array([r.time_ms for r in rs]) / 1e3, ok=stat != 'failure', nfev=np.array([r.nfev for r in rs]), wall=wall)
print(f'symbfitrs VarPro on the unclipped model: {wall:.1f} s wall; statuses {dict(zip(*np.unique(stat, return_counts=True)))}', flush=True)
print(f'  unconstrained solution infeasible: some of b,a1,a2 negative {neg.mean():.0%}; a width outside [0.1, 2pi] {out_w.mean():.0%}; an exponent outside [0.1, 4] {out_p.mean():.0%}; any {(neg | out_w | out_p).mean():.0%}')

# X4: VarPro + enumeration of the active set of the three non-negative linear parameters
parts = {'b': 'b', 'a1': f'a1*{PEAK1}', 'a2': f'a2*{PEAK2}'}
subsets = [s for k in (3, 2, 1) for s in itertools.combinations(('b', 'a1', 'a2'), k)]
models = {s: symbfitrs.compile(' + '.join(parts[k] for k in s)) for s in subsets}
best_loss = np.full(N, np.inf); Pnn = P0c.copy(); sub_used = [''] * N; any_feasible = np.zeros(N, bool); wall = 0.0
for s, m in models.items():
    t = time.time(); rs = run_pool(m, P0c); wall += time.time() - t
    for i, r in enumerate(rs):
        if not np.isfinite(r.loss): continue
        v = to_nb(r.params, m)
        lin_ok = all(v[nb.index(k)] >= -1e-9 for k in s)
        nl_ok = all(BOX[n][0] - 1e-6 <= v[nb.index(n)] <= BOX[n][1] + 1e-6 for n in m.params if n in BOX and n not in ('b', 'a1', 'a2', 'theta'))
        feas = lin_ok and nl_ok
        if (feas, -r.loss) > (any_feasible[i], -best_loss[i]) or (feas == any_feasible[i] and r.loss < best_loss[i]):
            if feas or not any_feasible[i]: best_loss[i], Pnn[i], sub_used[i], any_feasible[i] = r.loss, v, '+'.join(s), feas
res['symbfitrs VarPro + active-set enumeration'] = dict(P=Pnn, secs=np.full(N, wall / N), ok=np.ones(N, bool), nfev=np.zeros(N), feasible=any_feasible, wall=wall)
print(f'VarPro + enumeration over {len(subsets)} subsets of (b, a1, a2): {wall:.1f} s wall for {N} neurons; a feasible solution found in {any_feasible.mean():.0%}; '
      f'chosen subsets {dict(zip(*np.unique(sub_used, return_counts=True)))}', flush=True)

# ---- references from the earlier runs, then the unified scoring
res['symbfitrs LM, clipped model (original)'] = dict(P=P_s, secs=np.full(N, np.nan), ok=np.ones(N, bool), nfev=np.zeros(N))
res['Adam 15000 steps, clipped model'] = dict(P=P_a, secs=np.full(N, np.nan), ok=np.ones(N, bool), nfev=np.zeros(N))
names = list(res); L = np.array([score(res[k]['P'])[0] for k in names]); R2 = np.array([score(res[k]['P'])[1] for k in names])
best = np.nanmin(np.where(np.isfinite(L), L, np.inf), 0)
L_a = L[names.index('Adam 15000 steps, clipped model')]; L_s = L[names.index('symbfitrs LM, clipped model (original)')]
print(f'\nunified score (clipped notebook model, float64) vs the best result any method found per neuron ({N} neurons; {int(ab.sum())} are the Adam-better gap neurons):')
print(f'  {"method":46s}{"within 0.1%":>12s}{"within 1%":>11s}{"median over":>12s}{"gap neurons":>13s}{"Adam >1% lower":>16s}{"median R2":>10s}{"ms/fit":>8s}')
for k, l, r in zip(names, L, R2):
    sec = np.nanmedian(res[k]['secs']) * 1e3
    print(f'  {k:46s}{np.mean(l <= best * 1.001):12.0%}{np.mean(l <= best * 1.01):11.0%}{100 * (np.median(l / best) - 1):11.3f}%{np.mean((l <= L_a * 1.001)[ab]):13.0%}{np.mean(L_a < l * .99):16.0%}{np.median(r):10.4f}{sec:8.1f}')
np.savez(OUT, names=names, L=L, R2=R2, best=best, ab=ab, neg=neg, out_w=out_w, out_p=out_p, feas_enum=any_feasible, sub_used=sub_used, secs=np.array([res[k]['secs'] for k in names]),
         P=np.array([res[k]['P'] for k in names]))
print('saved', OUT)
