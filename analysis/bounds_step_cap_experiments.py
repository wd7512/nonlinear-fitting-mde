"""The Rust prototype: active-set bounds and a per-step cap on how far LM may move any parameter.
Same 300 neurons, same raw estimator start, same unified score (notebook's clipped model, float64) as before.
usage: PYTHONPATH=. python analysis/bounds_step_cap_experiments.py OUT.npz
"""
import sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from _neuro import *

OUT = sys.argv[1]
d = np.load('analysis/diagnosis.npz', allow_pickle=True); L_a, ab = d['L_a'], d['ab']
b0 = np.load('analysis/bounds_experiments.npz', allow_pickle=True); best0 = b0['best']
N = len(ROWS); cells = [trials(r) for r in ROWS]
TH = np.stack([c[0] for c in cells]); TRAIN = np.stack([c[1][0].mean(1) for c in cells]); TEST = np.stack([c[1][1].mean(1) for c in cells])
P0 = np.array([np.asarray(estimator(TH[i], TRAIN[i]), float) for i in range(N)])
nb = list(NAMES); TWO_PI = 2 * np.pi
BOUNDS = {'b': (0, None), 'a1': (0, None), 'a2': (0, None), **{k: (0.1, TWO_PI) for k in ('l1', 'r1', 'l2', 'r2')}, 'p1': (0.1, 4.0), 'p2': (0.1, 4.0)}
LO = np.array([0, 0, 0, 0, .1, .1, .1, .1, .1, .1]); HI = np.array([TWO_PI, np.inf, np.inf, np.inf] + [TWO_PI] * 4 + [4., 4.])   # the clip ranges (notebook order)
th64, y64, te64 = (torch.tensor(a, dtype=torch.float64) for a in (TH, TRAIN, TEST))
def score(P):
    with torch.no_grad():
        pr = gen_gauss(th64, torch.tensor(P, dtype=torch.float64)).squeeze(0)
        return ((pr - y64) ** 2).mean(1).numpy(), (1 - ((te64 - pr) ** 2).sum(1) / ((te64 - te64.mean(1, keepdim=True)) ** 2).sum(1)).numpy()

def one(i, bounds, cap, budget):
    return MODEL.fit(TH[i], TRAIN[i], p0=P0[i][ORDER], budget_ms=budget, trace=True, bounds=bounds, max_step=cap)

def run(bounds, cap, budget=3000.0):
    t = time.time()
    with ThreadPoolExecutor(4) as ex: rs = list(ex.map(lambda i: one(i, bounds, cap, budget), range(N)))
    P = np.array([r.params[INV] for r in rs])
    first_dead = []
    for r in rs:
        tr = r.trace[:, INV]; out = ((tr < LO - 1e-12) | (tr > HI + 1e-12)).any(1)
        first_dead.append(int(np.argmax(out)) if out.any() else -1)
    return dict(P=P, steps=np.array([r.steps for r in rs]), ms=np.array([r.time_ms for r in rs]), status=np.array([r.status for r in rs]),
                first_dead=np.array(first_dead), wall=time.time() - t)

settings = [('LM, clipped model (current symbfitrs)', None, None)]
settings += [(f'step cap {c:g}', None, c) for c in (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)]
settings += [('bounds (active-set)', BOUNDS, None)]
settings += [(f'bounds + step cap {c:g}', BOUNDS, c) for c in (1e-2, 3e-2, 0.1, 0.3, 1.0)]
res = {}
for lab, bnd, cap in settings:
    res[lab] = run(bnd, cap); r = res[lab]
    print(f'{lab:38s} {r["wall"]:5.1f} s wall | median steps {np.median(r["steps"]):6.0f}, median {np.median(r["ms"]):6.1f} ms | ' + ', '.join(f'{k} {np.mean(r["status"] == k):.0%}' for k in sorted(set(r['status']))), flush=True)

names = list(res); S = [score(res[k]['P']) for k in names]; L = np.array([s[0] for s in S]); R2 = np.array([s[1] for s in S])
best = np.minimum(best0, L.min(0)); alt = np.load('analysis/alternating_projection.npz', allow_pickle=True); best = np.minimum(best, alt['best'])
print(f'\nunified score, {N} neurons ({int(ab.sum())} Adam-better gap neurons); best = best result any method found so far')
print(f'  {"setting":38s}{"in 0.1%":>8s}{"in 1%":>7s}{"gap":>6s}{"Adam>1%":>9s}{"R2":>8s}{"steps":>8s}{"ms":>8s}{"dead-zone entry":>17s}{"at step 1":>10s}{"conv":>6s}')
for k, l, r2 in zip(names, L, R2):
    r = res[k]; fd = r['first_dead']
    print(f'  {k:38s}{np.mean(l <= best * 1.001):8.0%}{np.mean(l <= best * 1.01):7.0%}{np.mean((l <= L_a * 1.001)[ab]):6.0%}{np.mean(L_a < l * .99):9.0%}{np.median(r2):8.4f}'
          f'{np.median(r["steps"]):8.0f}{np.median(r["ms"]):8.1f}{np.mean(fd >= 0):17.0%}{np.mean(fd == 1):10.0%}{np.mean(r["status"] == "converged"):6.0%}')
np.savez(OUT, names=names, L=L, R2=R2, best=best, steps=np.array([res[k]['steps'] for k in names]), ms=np.array([res[k]['ms'] for k in names]),
         status=np.array([res[k]['status'] for k in names]), first_dead=np.array([res[k]['first_dead'] for k in names]), ab=ab, L_a=L_a)
print('saved', OUT)
