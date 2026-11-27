"""Why does Adam (15,000 steps) end with a lower training MSE than symbfitrs in a minority of neurons?

Checks, in order: (E1) is it a measurement artefact (Rust expression vs torch model)? (E2) which
parameters sit on the model's clips at each solution; (E3) gradient norm at each solution;
(E4) warm starts: LM from Adam's solution, Adam from LM's; (E5) overlap with multi-start basins;
(E6) how LM stops (its trace).
usage: PYTHONPATH=. python analysis/adam_vs_lm_diagnosis.py OUT.npz
"""
import sys, time
import numpy as np
import torch
from _neuro import *

OUT = sys.argv[1]
torch.set_num_threads(4)
est = lambda th, y: np.asarray(estimator(th, y), dtype=np.float32)
cells = [trials(r) for r in ROWS]
TH = np.stack([c[0] for c in cells]); TRAIN = np.stack([c[1][0].mean(1) for c in cells]); TEST = np.stack([c[1][1].mean(1) for c in cells])
N = len(ROWS)
P0 = np.array([est(TH[i], TRAIN[i]) for i in range(N)])
gg = lambda th, P: gen_gauss(th, P).squeeze(0)
th64, y64 = torch.tensor(TH, dtype=torch.float64), torch.tensor(TRAIN, dtype=torch.float64)
NAMES10 = ['theta', 'b', 'a1', 'a2', 'l1', 'r1', 'l2', 'r2', 'p1', 'p2']
LO = np.array([0, 0, 0, 0, .1, .1, .1, .1, .1, .1]); HI = np.array([2 * np.pi, np.inf, np.inf, np.inf] + [2 * np.pi] * 4 + [4., 4.])


def mse64(P, sel=slice(None)):
    with torch.no_grad():
        return ((gg(th64[sel], torch.tensor(P, dtype=torch.float64)) - y64[sel]) ** 2).mean(1).numpy()


def grad64(P):
    Pt = torch.tensor(P, dtype=torch.float64, requires_grad=True)
    ((gg(th64, Pt) - y64) ** 2).mean(1).sum().backward()
    return Pt.grad.numpy()


dead = lambda P: (np.asarray(P) < LO) | (np.asarray(P) > HI)       # strictly outside the clip range: zero gradient


def adam_run(th, Y, P0_, lr, steps, keep_hist=False):
    P = torch.tensor(P0_, dtype=torch.float32).requires_grad_(); opt = torch.optim.Adam([P], lr=lr)
    y = torch.tensor(Y, dtype=torch.float32); hist = torch.empty(steps, len(Y)) if keep_hist else None
    for s in range(steps):
        opt.zero_grad(); lc = ((gg(th, P) - y) ** 2).mean(1); (10 * lc.mean()).backward()
        if keep_hist: hist[s] = lc.detach()
        P.grad.nan_to_num_(0, 0, 0); opt.step()
    return P.detach().numpy().astype(np.float64), (None if hist is None else hist.numpy())


# ------------------------------------------------------------- the two fits being compared
t = time.time(); P_s, rs = fit_many(TH, TRAIN, P0, budget_ms=1000.0, trace=True); print(f'symbfitrs: {time.time() - t:.1f} s', flush=True)
sym_status = np.array([r.status for r in rs]); sym_steps = np.array([r.steps for r in rs]); sym_loss_rust = np.array([r.loss for r in rs])
t = time.time(); P_a, hist = adam_run(torch.tensor(TH, dtype=torch.float32), TRAIN, P0, 1e-3, 15000, keep_hist=True); print(f'Adam: {time.time() - t:.0f} s', flush=True)
L_s, L_a = mse64(P_s), mse64(P_a); ratio = L_a / L_s; ab = ratio < 0.99
print(f'\nAdam > 1% lower (torch-evaluated MSE): {ab.sum()} of {N}; > 0.1% lower: {(ratio < 0.999).sum()}; symbfitrs > 1% lower: {(ratio > 1.01).sum()}')

# ------------------------------------------------------------- E1: measurement artefact?
pred_r = np.array([MODEL.predict(TH[i], P_s[i][ORDER]) for i in range(N)])
with torch.no_grad(): pred_t = gg(th64, torch.tensor(P_s, dtype=torch.float64)).numpy()
dmax = np.abs(pred_r - pred_t).max(1); rel_loss = np.abs(sym_loss_rust - L_s) / L_s
print('\nE1 measurement check (symbfitrs solution evaluated by the Rust expression vs the torch model):')
print(f'  max |prediction difference| per neuron: median {np.median(dmax):.1e}, max {dmax.max():.1e}; relative loss mismatch: median {np.median(rel_loss):.1e}, max {rel_loss.max():.1e}')
print(f'  neurons with loss mismatch > 1e-6: {(rel_loss > 1e-6).sum()}, > 1e-3: {(rel_loss > 1e-3).sum()}')
print(f'  among the {ab.sum()} Adam-better neurons: median mismatch {np.median(rel_loss[ab]):.1e}, max {rel_loss[ab].max():.1e}')

# ------------------------------------------------------------- E2: parameters on the clips
D_s, D_a = dead(P_s), dead(P_a)
print('\nE2 parameters strictly outside their clip range (zero gradient there), per solution:')
print(f'  {"":24s}{"symbfitrs: any dead":>20s}{"Adam: any dead":>16s}')
for lab, m in (('Adam-better (>1%)', ab), ('all others', ~ab)):
    print(f'  {lab:24s}{D_s[m].any(1).mean():20.0%}{D_a[m].any(1).mean():16.0%}   (n={m.sum()})')
print('  by parameter, symbfitrs solution, Adam-better vs others: ' + ', '.join(f'{n} {D_s[ab][:, j].mean():.0%}/{D_s[~ab][:, j].mean():.0%}' for j, n in enumerate(NAMES10) if D_s[:, j].any()))
print('  by parameter, Adam solution,     Adam-better vs others: ' + ', '.join(f'{n} {D_a[ab][:, j].mean():.0%}/{D_a[~ab][:, j].mean():.0%}' for j, n in enumerate(NAMES10) if D_a[:, j].any()))

# ------------------------------------------------------------- E3: gradient norms at each solution (active parameters only)
gn = lambda P, D: np.sqrt(((grad64(P) * ~D) ** 2).sum(1))
g_s, g_a = gn(P_s, D_s), gn(P_a, D_a)
print('\nE3 L2 norm of the MSE gradient over active parameters:')
for lab, m in (('Adam-better', ab), ('others', ~ab)):
    print(f'  {lab:12s} symbfitrs median {np.median(g_s[m]):.1e} (p90 {np.percentile(g_s[m], 90):.1e}); Adam median {np.median(g_a[m]):.1e} (p90 {np.percentile(g_a[m], 90):.1e})')

# ------------------------------------------------------------- E4: warm starts
t = time.time(); P_la, rs_la = fit_many(TH, TRAIN, P_a, budget_ms=1000.0, trace=True); L_la = mse64(P_la)
print(f'\nE4 warm starts ({time.time() - t:.0f} s so far):')
print(f'  LM started from Adam\'s solution: moves it down by >0.1% in {np.mean(L_la < L_a * .999):.0%} of neurons (Adam-better: {np.mean((L_la < L_a * .999)[ab]):.0%}); '
      f'statuses {dict(zip(*np.unique([r.status for r in rs_la], return_counts=True)))}')
print(f'    median relative change {np.median(L_la / L_a - 1):+.1e}; ends above Adam by >0.1% in {np.mean(L_la > L_a * 1.001):.0%}')
rng = np.random.default_rng(3); ctrl = rng.choice(np.where(~ab)[0], ab.sum(), replace=False); sub = np.r_[np.where(ab)[0], ctrl]
P_as, _ = adam_run(torch.tensor(TH[sub], dtype=torch.float32), TRAIN[sub], P_s[sub].astype(np.float32), 1e-3, 5000)
L_as = mse64(P_as, sub)
print(f'  Adam (5000 steps) started from the LM solution, {len(sub)} neurons ({ab.sum()} Adam-better + {len(ctrl)} controls): '
      f'moves it down by >0.1% in {np.mean(L_as < L_s[sub] * .999):.0%} (Adam-better: {np.mean((L_as < L_s[sub] * .999)[:ab.sum()]):.0%}, controls: {np.mean((L_as < L_s[sub] * .999)[ab.sum():]):.0%});'
      f' median change Adam-better {np.median((L_as / L_s[sub] - 1)[:ab.sum()]):+.1e}')
reach = (L_as <= L_a[sub] * 1.01)[:ab.sum()]
print(f'    of the Adam-better neurons, Adam-from-LM reaches Adam-from-start\'s level (within 1%) in {reach.mean():.0%}')

# ------------------------------------------------------------- E5: multi-start basins (earlier experiment, same neurons and data)
ms = np.load('analysis/disc_multistart.npz'); best16 = np.nanmin(ms['loss'], 1)
print('\nE5 earlier 16-start symbfitrs runs (same train curves):')
print(f'  best of 16 starts is within 0.1% of or below Adam\'s final in {np.mean(best16 <= L_a * 1.001):.0%} of all neurons; '
      f'in {np.mean((best16 <= L_a * 1.001)[ab]):.0%} of the Adam-better ones (single estimator start: {np.mean((L_s <= L_a * 1.001)[ab]):.0%})')

# ------------------------------------------------------------- E6: how does LM stop?
def lm_loss(r, i):
    with torch.no_grad(): return ((gg(th64[i:i + 1].expand(len(r.trace), -1), torch.tensor(r.trace[:, INV], dtype=torch.float64)) - y64[i]) ** 2).mean(1).numpy()
traces = [lm_loss(r, i) for i, r in enumerate(rs)]
last_rel = np.array([(t[-2] - t[-1]) / t[-1] if len(t) > 1 else np.nan for t in traces])
first_dead = np.array([next((k for k in range(len(r.trace)) if dead(r.trace[k][INV][None]).any()), -1) for r in rs])
fin_loss_drop = np.array([(t[first_dead[i]] - t[-1]) / t[-1] if first_dead[i] >= 0 else np.nan for i, t in enumerate(traces)])
print('\nE6 LM trace:')
print(f'  relative loss decrease at the final accepted step: median {np.nanmedian(last_rel):.1e}; Adam-better {np.nanmedian(last_rel[ab]):.1e} vs others {np.nanmedian(last_rel[~ab]):.1e}')
print(f'  accepted steps: Adam-better median {np.median(sym_steps[ab]):.0f}, others {np.median(sym_steps[~ab]):.0f}; statuses (Adam-better): {dict(zip(*np.unique(sym_status[ab], return_counts=True)))}')
print(f'  a parameter enters a dead (clipped) region at some step in {np.mean(first_dead >= 0):.0%} of neurons ({np.mean(first_dead[ab] >= 0):.0%} of Adam-better); '
      f'after that, the loss still falls a further {np.nanmedian(fin_loss_drop):.1%} (median)')
np.savez(OUT, rows=ROWS, P_s=P_s, P_a=P_a, P0=P0, L_s=L_s, L_a=L_a, L_la=L_la, L_as=L_as, sub=sub, ab=ab, D_s=D_s, D_a=D_a, g_s=g_s, g_a=g_a,
         sym_status=sym_status, sym_steps=sym_steps, rel_loss=rel_loss, dmax=dmax, best16=best16, last_rel=last_rel, first_dead=first_dead,
         ex=np.where(ab)[0][np.argsort(ratio[ab])[:3]], hist_ab=hist[:, ab], traces_ab=np.array([traces[i] for i in np.where(ab)[0]], dtype=object), L_a_hist_idx=np.where(ab)[0])
print('saved', OUT)
