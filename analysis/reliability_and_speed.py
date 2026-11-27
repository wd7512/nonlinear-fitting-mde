"""Can symbfitrs give better / more reliable fits than Adam, and how much faster is it on a CPU?
A: objective quality at the end of the fit and how long Adam needs to match symbfitrs.
B: sensitivity to data scale and learning rate (Adam's step is absolute; LM is scale-free).
C: wall-clock vs number of neurons, same 4-core CPU.

usage: PYTHONPATH=. python analysis/reliability_and_speed.py OUT.json [QUICK]
"""
import json, sys, time
import numpy as np
import torch
from _neuro import *

OUT = sys.argv[1]; QUICK = len(sys.argv) > 2
STEPS = 300 if QUICK else 15000
torch.set_num_threads(4)
est = lambda th, y: np.asarray(estimator(th, y), dtype=np.float32)
rows = ROWS[:20] if QUICK else ROWS
cells = [trials(r) for r in rows]
TH = np.stack([c[0] for c in cells]); TRAIN = np.stack([c[1][0].mean(1) for c in cells]); TEST = np.stack([c[1][1].mean(1) for c in cells])
th_t = torch.tensor(TH, dtype=torch.float32)
gg = lambda th, P: gen_gauss(th, P).squeeze(0)       # per-cell angle grids, as in robustness.py


def adam_run(th, Y, P0, lr, steps, record=None):
    P = torch.tensor(P0).requires_grad_(); opt = torch.optim.Adam([P], lr=lr)
    y = torch.tensor(Y, dtype=torch.float32); hist = torch.empty(steps, len(Y))
    for s in range(steps):
        opt.zero_grad(); lc = ((gg(th, P) - y) ** 2).mean(1); (10 * lc.mean()).backward()
        hist[s] = lc.detach(); P.grad.nan_to_num_(0, 0, 0); opt.step()
        if record is not None: record(s, P)
    with torch.no_grad(): final = ((gg(th, P) - y) ** 2).mean(1).numpy()
    return P.detach().numpy(), hist.numpy(), final


def r2_cells(P, theta, Yt):
    with torch.no_grad(): pred = gg(torch.tensor(theta, dtype=torch.float32), torch.tensor(P, dtype=torch.float32)).numpy()
    return 1 - ((Yt - pred) ** 2).sum(1) / ((Yt - Yt.mean(1, keepdims=True)) ** 2).sum(1)


def run_both(scale, lr, label):
    """Adam (no early stopping, STEPS steps) and symbfitrs (to convergence) on the train repeat x scale."""
    Ytr, Yte = TRAIN * scale, TEST * scale
    P0 = np.array([est(TH[i], Ytr[i]) for i in range(len(rows))])
    t = time.time(); Pa, hist, fin_a = adam_run(th_t, Ytr, P0, lr, STEPS); ta = time.time() - t
    t = time.time(); Ps, rs = fit_many(TH, Ytr, P0, budget_ms=1000.0); ts = time.time() - t
    fin_s = np.array([r.loss for r in rs])
    t = time.time(); Ps100, rs100 = fit_many(TH, Ytr, P0, budget_ms=100.0); ts100 = time.time() - t
    rec = dict(label=label, scale=scale, lr=lr, adam_secs=ta, sym_secs=ts, sym100_secs=ts100, sym100_train=(np.array([r.loss for r in rs100]) / scale ** 2).tolist(),
               sym100_r2=r2_cells(np.nan_to_num(Ps100), TH, Yte).tolist(), sym100_status=[r.status for r in rs100],
               adam_train=(fin_a / scale ** 2).tolist(), sym_train=(fin_s / scale ** 2).tolist(),
               adam_r2=r2_cells(Pa, TH, Yte).tolist(), sym_r2=r2_cells(np.nan_to_num(Ps), TH, Yte).tolist(),
               sym_status=[r.status for r in rs], sym_steps=[r.steps for r in rs], sym_ms=[r.time_ms for r in rs],
               start_r2=r2_cells(P0, TH, Yte).tolist())
    print(f'{label:28s} Adam {ta:6.1f}s  symbfitrs {ts:5.1f}s | median train MSE/scale^2: Adam {np.median(rec["adam_train"]):.4f} '
          f'sym {np.median(rec["sym_train"]):.4f} ({ts100:.1f}s at 100 ms budget) | median test R2: Adam {np.median(rec["adam_r2"]):.3f} sym {np.median(rec["sym_r2"]):.3f}', flush=True)
    return rec, hist, fin_s / scale ** 2


out = dict(n=len(rows), steps=STEPS)
# ---------------------------------------------------------------- A: baseline setting + time-to-match
base, hist, fin_s = run_both(1.0, 1e-3, 'scale 1, lr 1e-3 (notebook)')
out['baseline'] = base
target = fin_s * 1.001 + 1e-12                                       # symbfitrs's final train MSE (+0.1%)
reached = hist <= target[None]
first = np.where(reached.any(0), reached.argmax(0) + 1, -1)
out['adam_steps_to_match_sym'] = first.tolist()
print(f'Adam matches symbfitrs\'s final train MSE (within 0.1%) in {np.mean(first > 0):.0%} of neurons within {STEPS} steps; '
      f'median step {np.median(first[first > 0]) if (first > 0).any() else float("nan"):.0f}', flush=True)
# ---------------------------------------------------------------- B: scale and learning-rate sensitivity
out['sens'] = []
for scale, lr, label in ((0.01, 1e-3, 'scale 0.01, lr 1e-3'), (100.0, 1e-3, 'scale 100, lr 1e-3'),
                         (1.0, 1e-4, 'scale 1, lr 1e-4'), (1.0, 1e-2, 'scale 1, lr 1e-2')):
    out['sens'].append(run_both(scale, lr, label)[0])
json.dump(out, open(OUT, 'w'))

# ---------------------------------------------------------------- C: timing vs number of neurons
from bench.suites import stringer2019  # noqa: F401  (only to fail early if the repo root is not on the path)
allrows = ROWS if QUICK else np.arange(len(X))
def passes(r):
    th, tr = trials(r); R = np.stack([t.mean(1) for t in tr]); n = R.shape[1]
    mu, var = R.mean(0), R.var(0, ddof=1); S2 = ((mu - mu.mean()) ** 2).mean() - (n - 1) / n**2 * (var / 2).sum(); V2 = (var / 2).sum() / n
    yc = np.clip(Y[r], 0, None); return S2 / (S2 + V2 + 1e-12) > .6 and abs((yc * np.exp(2j * X[r])).sum()) / yc.sum() > .3
passing = np.array([r for r in allrows if passes(r)]); print(f'{len(passing)} passing neurons for the timing study', flush=True)
timing = []
for n in ([10] if QUICK else [100, 300, 1000, len(passing)]):
    sel = passing[:n]; cs = [trials(r) for r in sel]
    TH_ = np.stack([c[0] for c in cs]); TR_ = np.stack([c[1][0].mean(1) for c in cs]); nb = TH_.shape[1]
    idx = np.random.default_rng(7).permutation(nb); nv = nb // 5; val2, val1, train = idx[:nv], idx[nv:2 * nv], idx[2 * nv:]
    P0a = np.array([est(TH_[i], TR_[i]) for i in range(n)]); P0t = np.array([est(TH_[i][train], TR_[i][train]) for i in range(n)])
    th_ = torch.tensor(TH_, dtype=torch.float32); y_ = torch.tensor(TR_, dtype=torch.float32)
    # Adam: per-step cost of the plain update and of the update + per-step validation scoring (the protocol), 200 steps
    def per_step(record):
        P = torch.tensor(P0t).requires_grad_(); opt = torch.optim.Adam([P], lr=1e-3); thb, yb = th_[:, train], y_[:, train]
        for s in range(210):
            if s == 10: t = time.time()
            opt.zero_grad(); (10 * ((gg(thb, P) - yb) ** 2).mean(1).mean()).backward(); P.grad.nan_to_num_(0, 0, 0); opt.step()
            if record:
                with torch.no_grad():
                    for b in (val1, val2): ((gg(th_[:, b], P) - y_[:, b]) ** 2).mean(1).numpy()
        return (time.time() - t) / 200
    plain, prot = per_step(False), per_step(True)
    adam_protocol = 15000 * (prot + plain)          # train phase with validation scoring + all-angle refit phase
    # symbfitrs: the same two phases, solver only, and with per-step trace scoring
    t = time.time()
    r1 = symbfitrs.Pool().fit_many([(MODEL, TH_[i][train], TR_[i][train], P0t[i][ORDER]) for i in range(n)], budget_ms=1000.0, progress=False, trace=True, max_steps=15000)
    r2_ = symbfitrs.Pool().fit_many([(MODEL, TH_[i], TR_[i], P0a[i][ORDER]) for i in range(n)], budget_ms=1000.0, progress=False)
    solve = time.time() - t
    t = time.time()
    symbfitrs.Pool().fit_many([(MODEL, TH_[i][train], TR_[i][train], P0t[i][ORDER]) for i in range(n)], budget_ms=100.0, progress=False, trace=True, max_steps=15000)
    symbfitrs.Pool().fit_many([(MODEL, TH_[i], TR_[i], P0a[i][ORDER]) for i in range(n)], budget_ms=100.0, progress=False)
    solve100 = time.time() - t
    t = time.time()
    for i, r in enumerate(r1):
        for b in (val1, val2): model_np(r.trace[:, INV], TH_[i][b])
    score = time.time() - t
    timing.append(dict(n=n, adam_plain_ms_step=plain * 1e3, adam_protocol_ms_step=prot * 1e3, adam_protocol_secs=adam_protocol,
                       sym_solve_secs=solve, sym_solve100_secs=solve100, sym_score_secs=score, sym_deadline=int(sum(r.status == 'deadline' for r in r1 + r2_)),
                       sym_median_ms=float(np.median([r.time_ms for r in r1 + r2_]))))
    print(f'n={n:5d}: Adam protocol (extrapolated from 200 steps x 15000, CPU) {adam_protocol:7.0f} s | symbfitrs solve {solve:5.1f} s '
          f'(+{score:.1f} s scoring traces; {solve100:.1f} s at a 100 ms budget) | {timing[-1]["sym_deadline"]} of {2 * n} fits hit the 1 s budget', flush=True)
out['timing'] = timing
json.dump(out, open(OUT, 'w')); print('saved', OUT)
