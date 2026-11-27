"""Control for bounds_experiments.py: is the benefit of true bounds just the feasible start?
Unbounded solvers on the clipped model, started from the same feasible point the bounded SciPy runs used."""
import time
import numpy as np, torch
from scipy.optimize import least_squares
from _neuro import *
d = np.load('analysis/diagnosis.npz', allow_pickle=True); b = np.load('analysis/bounds_experiments.npz', allow_pickle=True)
ab, L_a = d['ab'], d['L_a']; names0 = [str(n) for n in b['names']]; L0 = b['L']; best0 = b['best']
N = len(ROWS); cells = [trials(r) for r in ROWS]
TH = np.stack([c[0] for c in cells]); TRAIN = np.stack([c[1][0].mean(1) for c in cells]); TEST = np.stack([c[1][1].mean(1) for c in cells])
nb = list(NAMES); TWO_PI = 2 * np.pi
LO = np.array([-np.inf, 0, 0, 0, .1, .1, .1, .1, .1, .1]); HI = np.array([np.inf, np.inf, np.inf, np.inf] + [TWO_PI] * 4 + [4., 4.])
P0 = np.array([np.asarray(estimator(TH[i], TRAIN[i]), float) for i in range(N)])
P0c = np.clip(P0, LO, HI); P0c[:, 0] = np.mod(P0c[:, 0], TWO_PI)
print(f'raw estimator start has a parameter outside its range in {np.mean(((P0 < LO) | (P0 > HI)).any(1)):.0%} of neurons; the feasible start moves it to the boundary')
th64, y64 = torch.tensor(TH, dtype=torch.float64), torch.tensor(TRAIN, dtype=torch.float64)
def score(P):
    with torch.no_grad(): return ((gen_gauss(th64, torch.tensor(P, dtype=torch.float64)).squeeze(0) - y64) ** 2).mean(1).numpy()
out = {}
for m in ('lm', 'trf', 'dogbox'):
    P = np.zeros((N, 10)); t = time.perf_counter()
    for i in range(N):
        f = lambda x: MODEL.predict(TH[i], x) - TRAIN[i]
        try: x = least_squares(f, P0c[i][ORDER], method=m).x
        except Exception: x = P0c[i][ORDER]
        P[i] = x[INV]
    out[f'SciPy {m}, clipped model, FEASIBLE start'] = (score(P), (time.perf_counter() - t) / N * 1e3)
P, rs = fit_many(TH, TRAIN, P0c, budget_ms=1000.0)
out['symbfitrs LM, clipped model, FEASIBLE start'] = (score(P), np.median([r.time_ms for r in rs]))
best = np.minimum(best0, np.min([v[0] for v in out.values()], 0))
print(f'{"method":48s}{"within 0.1% of best":>20s}{"within 1%":>11s}{"gap neurons":>13s}{"Adam >1% lower":>16s}{"ms/fit":>8s}')
for k, (l, ms) in out.items():
    print(f'{k:48s}{np.mean(l <= best * 1.001):20.0%}{np.mean(l <= best * 1.01):11.0%}{np.mean((l <= L_a * 1.001)[ab]):13.0%}{np.mean(L_a < l * .99):16.0%}{ms:8.1f}')
for k in ('SciPy lm, clipped model, no bounds', 'SciPy trf, clipped model, no bounds', 'SciPy dogbox, clipped model, no bounds', 'SciPy trf, true bounds', 'SciPy dogbox, true bounds', 'symbfitrs LM, clipped model (original)'):
    l = L0[names0.index(k)]
    print(f'{"(earlier) " + k:48s}{np.mean(l <= best * 1.001):20.0%}{np.mean(l <= best * 1.01):11.0%}{np.mean((l <= L_a * 1.001)[ab]):13.0%}{np.mean(L_a < l * .99):16.0%}')
np.savez('analysis/bounds_controls.npz', names=list(out), L=np.array([v[0] for v in out.values()]))
