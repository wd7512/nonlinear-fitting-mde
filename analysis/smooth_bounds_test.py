"""Does removing the clip-induced dead zones (smooth reparametrisation) remove Adam's advantage?
Bounds become: b, a = u^2 (>= 0); width = 0.1 + u^2 (>= 0.1, no 2*pi cap); p = 0.1 + 3.9*sigmoid(u) in (0.1, 4)."""
import time
import numpy as np
import symbfitrs
from _neuro import *

d = np.load('analysis/diagnosis.npz', allow_pickle=True)
P0, P_s, P_a, ab, L_s, L_a = d['P0'].astype(float), d['P_s'], d['P_a'], d['ab'], d['L_s'], d['L_a']
N = len(ROWS); cells = [trials(r) for r in ROWS]
TH = np.stack([c[0] for c in cells]); TRAIN = np.stack([c[1][0].mean(1) for c in cells]); TEST = np.stack([c[1][1].mean(1) for c in cells])
e = EXPR; W = '6.283185307179586'
for n in ('b', 'a1', 'a2'):
    assert f'max({n}, 0)' in e; e = e.replace(f'max({n}, 0)', f'(u{n}*u{n})')
for n in ('l1', 'r1', 'l2', 'r2'):
    s = f'min(max({n}, 0.1), {W})'; assert s in e; e = e.replace(s, f'(0.1 + u{n}*u{n})')
for n in ('p1', 'p2'):
    s = f'min(max({n}, 0.1), 4.0)'; assert s in e; e = e.replace(s, f'(0.1 + 3.9/(1 + exp(-u{n})))')
M2 = symbfitrs.compile(e); print('reparametrised model parameters:', M2.params)
U0 = np.zeros((N, 10))
U0[:, 0] = np.clip(P0[:, 0], 0, 2 * np.pi)
U0[:, 1:4] = np.sqrt(np.maximum(P0[:, 1:4], 1e-4))
U0[:, 4:8] = np.sqrt(np.maximum(P0[:, 4:8] - 0.1, 1e-4))
pp = np.clip(P0[:, 8:10], 0.11, 3.99); U0[:, 8:10] = np.log((pp - 0.1) / (4.0 - pp))
names_u = ['theta', 'ub', 'ua1', 'ua2', 'ul1', 'ur1', 'ul2', 'ur2', 'up1', 'up2']
ordr = [names_u.index(p) for p in M2.params]; inv = np.argsort(ordr)
t = time.time()
rs = symbfitrs.Pool().fit_many([(M2, TH[i], TRAIN[i], U0[i][ordr]) for i in range(N)], budget_ms=1000.0, progress=False)
secs = time.time() - t
U = np.array([r.params[inv] for r in rs]); L_new = np.array([r.loss for r in rs])
eff = np.c_[U[:, 0], U[:, 1:4] ** 2, 0.1 + U[:, 4:8] ** 2, 0.1 + 3.9 / (1 + np.exp(-U[:, 8:10]))]   # effective parameters
wide = (eff[:, 4:8] > 2 * np.pi).any(1)
pred = np.array([M2.predict(TH[i], U[i][ordr]) for i in range(N)])
r2 = lambda y, p: 1 - ((y - p) ** 2).sum(1) / ((y - y.mean(1, keepdims=True)) ** 2).sum(1)
R2_new = r2(TEST, pred)
with np.errstate(all='ignore'):
    import torch
    pa = gen_gauss(torch.tensor(TH, dtype=torch.float32)[0:1], torch.tensor(P_a[:1], dtype=torch.float32))  # (shape check only)
th_t = torch.tensor(TH, dtype=torch.float64)
R2_a = r2(TEST, gen_gauss(th_t, torch.tensor(P_a, dtype=torch.float64)).squeeze(0).detach().numpy() if False else np.array([gen_gauss(th_t[i], torch.tensor(P_a[i:i + 1], dtype=torch.float64))[0].detach().numpy() for i in range(N)]))
R2_s = r2(TEST, np.array([gen_gauss(th_t[i], torch.tensor(P_s[i:i + 1], dtype=torch.float64))[0].detach().numpy() for i in range(N)]))
st = np.array([r.status for r in rs])
print(f'LM on the smooth-bounds model: {secs:.1f} s wall for {N} neurons; statuses {dict(zip(*np.unique(st, return_counts=True)))}; median accepted steps {np.median([r.steps for r in rs]):.0f}')
print(f'neurons whose effective width exceeds 2*pi (the model is wider than the original there): {wide.sum()}')
print('\nfinal train MSE relative to Adam (15000 steps), the share of neurons where Adam is lower:')
for lab, L in (('clipped model, LM (original)', L_s), ('smooth-bounds model, LM', L_new)):
    print(f'  {lab:32s} Adam >1% lower: {np.mean(L_a < L * .99):4.0%}   >0.1% lower: {np.mean(L_a < L * .999):4.0%}   LM >1% lower: {np.mean(L < L_a * .99):4.0%}   median LM/Adam {np.median(L / L_a):.4f}')
print(f'  smooth vs clipped LM:        smooth lower by >0.1% in {np.mean(L_new < L_s * .999):.0%} of neurons, >1% in {np.mean(L_new < L_s * .99):.0%}; clipped lower by >1% in {np.mean(L_s < L_new * .99):.0%}')
print(f'  the {ab.sum()} Adam-better neurons: smooth LM reaches Adam\'s level (within 1%) in {np.mean((L_new <= L_a * 1.01)[ab]):.0%} (clipped LM 0%); excluding wide-width neurons: {np.mean((L_new <= L_a * 1.01)[ab & ~wide]):.0%} of {(ab & ~wide).sum()}')
print(f'median held-out R2: Adam {np.median(R2_a):.4f}, clipped LM {np.median(R2_s):.4f}, smooth-bounds LM {np.median(R2_new):.4f}')
dd = R2_new - R2_a; print(f'  smooth LM minus Adam: median {np.median(dd):+.4f}; in the Adam-better neurons {np.median(dd[ab]):+.4f}')
np.savez('analysis/smooth_bounds.npz', L_new=L_new, R2_new=R2_new, wide=wide, eff=eff, status=st, steps=[r.steps for r in rs], ms=[r.time_ms for r in rs])
