"""F4: Rust active-set bounds on the UNCLIPPED expression (bounds replace the clips). PYTHONPATH=. python analysis/bounds_unclipped_rust.py"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, symbfitrs
from _neuro import *
d = np.load('analysis/diagnosis.npz', allow_pickle=True); L_a, ab = d['L_a'], d['ab']
c = np.load('analysis/bounds_step_cap.npz', allow_pickle=True); best = c['best']; names = [str(n) for n in c['names']]; Lc = c['L']
N = len(ROWS); cells = [trials(r) for r in ROWS]
TH = np.stack([x[0] for x in cells]); TRAIN = np.stack([x[1][0].mean(1) for x in cells]); TWO_PI = 2 * np.pi; nb = list(NAMES)
P0 = np.array([np.asarray(estimator(TH[i], TRAIN[i]), float) for i in range(N)])
e = EXPR; W = '6.283185307179586'
e = e.replace(f'min(max(theta, 0), {W})', 'theta')
for n in ('l1', 'r1', 'l2', 'r2'): e = e.replace(f'min(max({n}, 0.1), {W})', n)
for n in ('p1', 'p2'): e = e.replace(f'min(max({n}, 0.1), 4.0)', n)
i1, i2 = e.index(' + max(a1, 0)*'), e.index(' + max(a2, 0)*')
FREE = symbfitrs.compile(f'b + a1*{e[i1 + len(" + max(a1, 0)*"):i2]} + a2*{e[i2 + len(" + max(a2, 0)*"):]}')
order = [nb.index(p) for p in FREE.params]; inv = np.argsort(order)
BOUNDS = {'b': (0, None), 'a1': (0, None), 'a2': (0, None), **{k: (0.1, TWO_PI) for k in ('l1', 'r1', 'l2', 'r2')}, 'p1': (0.1, 4.0), 'p2': (0.1, 4.0)}
th64, y64 = torch.tensor(TH, dtype=torch.float64), torch.tensor(TRAIN, dtype=torch.float64)
def score(P):
    with torch.no_grad(): return ((gen_gauss(th64, torch.tensor(P, dtype=torch.float64)).squeeze(0) - y64) ** 2).mean(1).numpy()
print(f'{"setting":46s}{"in 0.1%":>8s}{"in 1%":>7s}{"gap":>6s}{"Adam>1%":>9s}{"steps":>8s}{"ms":>8s}{"conv":>6s}  statuses')
out = {}
for lab, bnd, cap in (('unclipped model, bounds', BOUNDS, None), ('unclipped model, bounds + cap 0.03', BOUNDS, 0.03), ('unclipped model, bounds + cap 0.1', BOUNDS, 0.1), ('unclipped model, bounds + cap 0.3', BOUNDS, 0.3), ('unclipped model, bounds + cap 1.0', BOUNDS, 1.0)):
    with ThreadPoolExecutor(4) as ex:
        rs = list(ex.map(lambda i: FREE.fit(TH[i], TRAIN[i], p0=P0[i][order], budget_ms=3000.0, bounds=bnd, max_step=cap), range(N)))
    P = np.array([r.params[inv] for r in rs]); L = score(P); st = np.array([r.status for r in rs]); out[lab] = L
    print(f'{lab:46s}{np.mean(L <= best * 1.001):8.0%}{np.mean(L <= best * 1.01):7.0%}{np.mean((L <= L_a * 1.001)[ab]):6.0%}{np.mean(L_a < L * .99):9.0%}{np.median([r.steps for r in rs]):8.0f}{np.median([r.time_ms for r in rs]):8.1f}{np.mean(st == "converged"):6.0%}  {dict(zip(*np.unique(st, return_counts=True)))}')
np.savez('analysis/bounds_unclipped_rust.npz', names=list(out), L=np.array(list(out.values())))
