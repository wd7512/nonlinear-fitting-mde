"""Follow-ups to bounds_step_cap_experiments.py: (F1) bound theta too; (F2) are the 'stalled' tiny-cap fits at the optimum?;
(F3) is the absolute cap scale-sensitive?  PYTHONPATH=. python analysis/bounds_step_cap_followups.py"""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
from _neuro import *
d = np.load('analysis/diagnosis.npz', allow_pickle=True); L_a, ab = d['L_a'], d['ab']
c = np.load('analysis/bounds_step_cap.npz', allow_pickle=True); names = [str(n) for n in c['names']]; Lc, best, status = c['L'], c['best'], c['status']
N = len(ROWS); cells = [trials(r) for r in ROWS]
TH = np.stack([x[0] for x in cells]); TRAIN = np.stack([x[1][0].mean(1) for x in cells])
P0 = np.array([np.asarray(estimator(TH[i], TRAIN[i]), float) for i in range(N)]); TWO_PI = 2 * np.pi
BOX = {'b': (0, None), 'a1': (0, None), 'a2': (0, None), **{k: (0.1, TWO_PI) for k in ('l1', 'r1', 'l2', 'r2')}, 'p1': (0.1, 4.0), 'p2': (0.1, 4.0)}
LO = np.array([0, 0, 0, 0, .1, .1, .1, .1, .1, .1]); HI = np.array([TWO_PI, np.inf, np.inf, np.inf] + [TWO_PI] * 4 + [4., 4.])
th64, y64 = torch.tensor(TH, dtype=torch.float64), torch.tensor(TRAIN, dtype=torch.float64)
def score(P):
    with torch.no_grad(): return ((gen_gauss(th64, torch.tensor(P, dtype=torch.float64)).squeeze(0) - y64) ** 2).mean(1).numpy()
def run(bounds, cap, scale=1.0, budget=3000.0):
    sc = np.array([1, scale, scale, scale, 1, 1, 1, 1, 1, 1.0])
    with ThreadPoolExecutor(4) as ex:
        rs = list(ex.map(lambda i: MODEL.fit(TH[i], TRAIN[i] * scale, p0=(P0[i] * sc)[ORDER], budget_ms=budget, trace=True, bounds=bounds, max_step=cap), range(N)))
    P = np.array([r.params[INV] for r in rs]) / sc
    fd = []
    for r in rs:
        tr = r.trace[:, INV] / sc; o = ((tr < LO - 1e-12) | (tr > HI + 1e-12)).any(1); fd.append(int(np.argmax(o)) if o.any() else -1)
    return score(P), np.array([r.steps for r in rs]), np.array([r.time_ms for r in rs]), np.array([r.status for r in rs]), np.array(fd)
def line(lab, L, steps, ms, st, fd):
    print(f'  {lab:42s}{np.mean(L <= best * 1.001):7.0%}{np.mean(L <= best * 1.01):7.0%}{np.mean((L <= L_a * 1.001)[ab]):6.0%}{np.mean(L_a < L * .99):9.0%}{np.median(steps):8.0f}{np.median(ms):8.1f}{np.mean(fd >= 0):9.0%}{np.mean(fd == 1):8.0%}{np.mean(st == "converged"):6.0%}')
hdr = f'  {"setting":42s}{"in 0.1%":>7s}{"in 1%":>7s}{"gap":>6s}{"Adam>1%":>9s}{"steps":>8s}{"ms":>8s}{"dead":>9s}{"@step1":>8s}{"conv":>6s}'
# F1
print('F1 bounds that also include theta in [0, 2pi] (the notebook model clips theta too):'); print(hdr)
B2 = {**BOX, 'theta': (0, TWO_PI)}
for lab, cap in (('bounds incl. theta', None), ('bounds incl. theta + cap 0.03', 0.03), ('bounds incl. theta + cap 0.1', 0.1), ('bounds incl. theta + cap 0.3', 0.3)):
    line(lab, *run(B2, cap))
# F2
print('\nF2 tiny caps: are the "stalled" fits already at the best result? (within 0.1% of best / of the same neuron\'s uncapped LM result)')
for k in ('step cap 0.001', 'step cap 0.003', 'step cap 0.01'):
    j = names.index(k); s = status[j]
    for stt in sorted(set(s)):
        m = s == stt; print(f'  {k:16s} {stt:10s} n={m.sum():3d}: within 0.1% of best {np.mean(Lc[j][m] <= best[m] * 1.001):4.0%}; median loss vs uncapped LM {np.median(Lc[j][m] / Lc[0][m] - 1):+.2e}')
# F3
print('\nF3 data rescaled by s (parameters b, a1, a2 scale with it; the cap stays an absolute number):'); print(hdr)
for s in (0.01, 100.0):
    for lab, cap in (('no cap', None), ('cap 0.03', 0.03), ('cap 0.1', 0.1)):
        line(f'scale {s:g}, {lab}', *run(None, cap, scale=s))
