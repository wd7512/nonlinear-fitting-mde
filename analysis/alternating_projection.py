"""Prototype of the "alternate unconstrained and bound-fixed steps" idea, in a plain NumPy LM loop.

scheme 'alternate' (the idea): odd iterations take a full LM step on all parameters; the point is then
projected into the box (anything outside goes onto the bound); even iterations take an LM step on the
parameters NOT sitting on a bound (the ones on a bound are held fixed); then the bounded parameters are
released again at the next odd iteration.
Controls: 'none' (unbounded, clamp at the end), 'project_each' (project after every step), 'alternate_literal'
(accept the unbounded step on its own, unprojected loss), 'alternate4' (release only every 4th step),
'active_set' (hold a parameter at a bound only while the gradient pushes outward; MPFIT-style).
usage: PYTHONPATH=. python analysis/alternating_projection.py OUT.npz [N_QUICK]
"""
import sys, time
import numpy as np
import torch
import symbfitrs
from _neuro import *

OUT = sys.argv[1]; NQ = int(sys.argv[2]) if len(sys.argv) > 2 else None
b = np.load('analysis/bounds_experiments.npz', allow_pickle=True); names0 = [str(n) for n in b['names']]; L0, best0, ab = b['L'], b['best'], b['ab']
d = np.load('analysis/diagnosis.npz', allow_pickle=True); L_a = d['L_a']
N = NQ or len(ROWS); cells = [trials(r) for r in ROWS[:N]]
TH = np.stack([c[0] for c in cells]); TRAIN = np.stack([c[1][0].mean(1) for c in cells]); TEST = np.stack([c[1][1].mean(1) for c in cells])
nb = list(NAMES); TWO_PI = 2 * np.pi
BOX = {'theta': (-np.inf, np.inf), 'b': (0, np.inf), 'a1': (0, np.inf), 'a2': (0, np.inf), **{k: (0.1, TWO_PI) for k in ('l1', 'r1', 'l2', 'r2')}, 'p1': (0.1, 4.0), 'p2': (0.1, 4.0)}
e = EXPR; W = '6.283185307179586'
e = e.replace(f'min(max(theta, 0), {W})', 'theta')
for n in ('l1', 'r1', 'l2', 'r2'): e = e.replace(f'min(max({n}, 0.1), {W})', n)
for n in ('p1', 'p2'): e = e.replace(f'min(max({n}, 0.1), 4.0)', n)
i1, i2 = e.index(' + max(a1, 0)*'), e.index(' + max(a2, 0)*')
FREE = symbfitrs.compile(f'b + a1*{e[i1 + len(" + max(a1, 0)*"):i2]} + a2*{e[i2 + len(" + max(a2, 0)*"):]}')
LO = np.array([BOX[n][0] for n in FREE.params]); HI = np.array([BOX[n][1] for n in FREE.params])
to_nb = lambda x: np.array([x[FREE.params.index(n)] for n in nb]); from_nb = lambda v: np.array([v[nb.index(n)] for n in FREE.params])
P0 = np.array([np.asarray(estimator(TH[i], TRAIN[i]), float) for i in range(N)])
P0c = np.clip(P0, [BOX[n][0] for n in nb], [BOX[n][1] for n in nb]); P0c[:, 0] = np.mod(P0c[:, 0], TWO_PI)
th64, y64, te64 = (torch.tensor(a, dtype=torch.float64) for a in (TH, TRAIN, TEST))
def score(P):
    with torch.no_grad():
        pr = gen_gauss(th64, torch.tensor(P, dtype=torch.float64)).squeeze(0)
        return ((pr - y64) ** 2).mean(1).numpy(), (1 - ((te64 - pr) ** 2).sum(1) / ((te64 - te64.mean(1, keepdim=True)) ** 2).sum(1)).numpy()

EPS = 1e-12
def lm(th, y, x0, scheme, max_iter=300, tau=1e-3):
    p = len(x0); clip = lambda v: np.clip(v, LO, HI)
    res = lambda v: FREE.predict(th, v) - y
    x = clip(x0); r = res(x); sse = float(r @ r); nfev = 1
    best_x, best_sse = x.copy(), sse
    d = np.zeros(p); lam = None; nu = 2.0; small = 0; its = 0
    for it in range(max_iter):
        its = it + 1
        # forward-difference Jacobian (one-sided, stepping away from a bound it sits on)
        J = np.empty((len(y), p))
        for j in range(p):
            h = np.sqrt(np.finfo(float).eps) * max(1.0, abs(x[j])); s = -1.0 if x[j] >= HI[j] - EPS else 1.0
            xj = x.copy(); xj[j] += s * h; J[:, j] = (res(xj) - r) / (s * h)
        nfev += p
        if not np.all(np.isfinite(J)): break
        cn = np.linalg.norm(J, axis=0); d = np.maximum(d, np.where(cn > 0, cn, 1.0))
        at = (x <= LO + EPS) | (x >= HI - EPS)
        full = np.ones(p, bool)
        if scheme in ('alternate', 'alternate_literal'): free = full if it % 2 == 0 else ~at
        elif scheme == 'alternate4': free = full if it % 4 == 0 else ~at
        elif scheme == 'active_set':
            g = J.T @ r; free = ~(((x <= LO + EPS) & (g > 0)) | ((x >= HI - EPS) & (g < 0)))
        else: free = full
        if not free.any(): free = full
        raw_accept = scheme in ('none', 'alternate_literal') and (free.all() or scheme == 'none')   # accept on the unprojected loss
        Js = J[:, free] / d[free]; nf = int(free.sum())
        if lam is None: lam = tau * np.linalg.svd(Js, compute_uv=False)[0] ** 2
        accepted = False
        for _ in range(40):
            A = np.vstack([Js, np.sqrt(lam) * np.eye(nf)]); bvec = np.r_[-r, np.zeros(nf)]
            ds = np.linalg.lstsq(A, bvec, rcond=None)[0]; step = np.zeros(p); step[free] = ds / d[free]
            xt = x + step
            xe = xt if raw_accept else clip(xt)
            with np.errstate(all='ignore'): rt = res(xe)
            nfev += 1; sse_t = float(rt @ rt)
            pred = sse - float(np.sum((r + Js @ ds) ** 2))
            if np.isfinite(sse_t) and sse_t < sse:
                rho = (sse - sse_t) / pred if pred > 0 else 1.0
                lam *= max(1 / 3, 1 - (2 * rho - 1) ** 3); nu = 2.0; accepted = True; break
            lam *= nu; nu *= 2
            if not np.isfinite(lam) or lam > 1e32: break
        if not accepted: small += 1;
        if not accepted and (lam > 1e32 or not np.isfinite(lam)): break
        if accepted:
            actual = sse - sse_t
            x = xe if scheme == 'none' else clip(xt)
            if scheme == 'none' or not (scheme == 'alternate_literal' and free.all()): r, sse = rt, sse_t
            else: r = res(x); sse = float(r @ r); nfev += 1          # literal: the stored point is the projected one
            if sse < best_sse: best_x, best_sse = x.copy(), sse
            small = small + 1 if actual <= 1e-12 * max(sse, 1e-300) else 0
        if small >= 2: break
    out = clip(best_x) if scheme != 'none' else clip(x)
    return out, its, nfev, small >= 2

schemes = ['none', 'project_each', 'alternate', 'alternate_literal', 'alternate4', 'active_set']
labels = {'none': 'LM on unclipped model, clamp at the end', 'project_each': 'project into the box after every step',
          'alternate': 'ALTERNATE (your idea): full step, project, step on non-pinned, release', 'alternate_literal': 'alternate, literal (accept the unbounded step on its own loss)',
          'alternate4': 'alternate, release only every 4th step', 'active_set': 'gradient-based active set (MPFIT-style)'}
rec = {}
for sc in schemes:
    t = time.time(); P = np.zeros((N, 10)); its = np.zeros(N); nfe = np.zeros(N); conv = np.zeros(N, bool)
    for i in range(N):
        x, its[i], nfe[i], conv[i] = lm(TH[i], TRAIN[i], from_nb(P0c[i]), sc); P[i] = to_nb(x)
    rec[sc] = dict(P=P, its=its, nfev=nfe, conv=conv, secs=time.time() - t)
    print(f'{sc:18s} done in {time.time() - t:5.1f} s; median iterations {np.median(its):.0f}, converged {conv.mean():.0%}', flush=True)

# unified scoring against the best known result (earlier methods plus these)
Ls = {sc: score(rec[sc]['P']) for sc in schemes}
best = np.minimum(best0[:N], np.min([Ls[sc][0] for sc in schemes], 0)); La = L_a[:N]; abn = ab[:N]
def row(lab, l, r2, its=None, conv=None):
    print(f'  {lab:72s}{np.mean(l <= best * 1.001):9.0%}{np.mean(l <= best * 1.01):8.0%}{np.mean((l <= La * 1.001)[abn]):8.0%}{np.mean(La < l * .99):8.0%}{np.median(r2):9.4f}' + ('' if its is None else f'{np.median(its):8.0f}{np.mean(conv):7.0%}'))
print(f'\nunified score, {N} neurons ({int(abn.sum())} Adam-better gap neurons); iterations are LM iterations (one step each)')
print(f'  {"method":72s}{"in 0.1%":>9s}{"in 1%":>8s}{"gap":>8s}{"Adam>1%":>9s}{"R2":>9s}{"iters":>8s}{"conv":>7s}')
for sc in schemes: row(labels[sc], *Ls[sc], rec[sc]['its'], rec[sc]['conv'])
print('  --- earlier results, same neurons')
for k in ('Adam 15000 steps, clipped model', 'symbfitrs LM, clipped model (original)', 'SciPy lm, clipped model, no bounds', 'SciPy trf, true bounds', 'SciPy dogbox, true bounds'):
    j = names0.index(k); row(k, L0[j][:N], b['R2'][j][:N])
np.savez(OUT, schemes=schemes, L=np.array([Ls[s][0] for s in schemes]), R2=np.array([Ls[s][1] for s in schemes]), best=best,
         its=np.array([rec[s]['its'] for s in schemes]), conv=np.array([rec[s]['conv'] for s in schemes]), nfev=np.array([rec[s]['nfev'] for s in schemes]),
         P=np.array([rec[s]['P'] for s in schemes]))
print('saved', OUT)
