"""How well is the peakiness exponent p identified? (1) trial-level bootstrap of every neuron's
fit; (2) synthetic recovery with known p, matched noise, and varying trials per bin.

usage: python analysis/exponent_identifiability.py OUT_PREFIX [B]
"""
import sys, time
import numpy as np
from _neuro import *

OUT = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 100
rng = np.random.default_rng(1)
est = lambda th, y: np.asarray(estimator(th, y), dtype=np.float32)

# ------------------------------------------------------------------ (1) bootstrap on real data
t0 = time.time()
cells = [trials(r) for r in ROWS]
P0 = np.array([est(th, tr[0].mean(1)) for th, tr in cells])
P_hat, res = fit_many([c[0] for c in cells], [c[1][0].mean(1) for c in cells], P0)            # even half
P_odd, _ = fit_many([c[0] for c in cells], [c[1][1].mean(1) for c in cells], np.array([est(c[0], c[1][1].mean(1)) for c in cells]))
boot_p = np.full((len(ROWS), B, 2), np.nan); boot_amp = np.full((len(ROWS), B, 2), np.nan)
n_deadline = 0; n_fits = 0
for b in range(B):
    curves, p0s = [], []
    for th, tr in cells:
        n = len(th); idx = rng.integers(0, tr[0].shape[1], (n, tr[0].shape[1]))
        c = np.take_along_axis(tr[0], idx, 1).mean(1); curves.append(c); p0s.append(est(th, c))
    P, rs = fit_many([c[0] for c in cells], curves, np.array(p0s))
    boot_p[:, b] = clipped_p(P); boot_amp[:, b] = np.clip(P[:, 2:4], 0, None)
    n_deadline += sum(r.status == 'deadline' for r in rs); n_fits += len(rs)
print(f'bootstrap: {n_fits} fits in {time.time() - t0:.0f} s wall ({n_deadline} hit the 100 ms budget)', flush=True)
np.savez(OUT + '_boot.npz', rows=ROWS, p_hat=clipped_p(P_hat), p_odd=clipped_p(P_odd), amp_hat=np.clip(P_hat[:, 2:4], 0, None),
         amp_odd=np.clip(P_odd[:, 2:4], 0, None), theta_hat=P_hat[:, 0], theta_odd=P_odd[:, 0], p_start=clipped_p(P0),
         boot_p=boot_p, boot_amp=boot_amp, secs=time.time() - t0, n_fits=n_fits, n_deadline=n_deadline)

# ------------------------------------------------------------------ (2) synthetic recovery
t0 = time.time()
P_TRUE, N_TR, REP = [0.6, 1.0, 1.5, 2.0, 3.0], [4, 8, 16, 32, 64], 8
sub = list(range(0, len(ROWS), 3))                  # 100 neurons give the realistic geometry and noise level
rows = []
for ci in sub:
    th, tr = cells[ci]; P = P_hat[ci].copy()
    sd = float(np.std(tr[0] - model_np(P[None], th)[0][:, None]))     # per-trial noise SD in normalised units
    for pt in P_TRUE:
        T = P.copy(); T[2] = max(T[2], 1e-3); T[3] = max(T[3], 0.5 * T[2]); T[8:10] = pt   # both peaks real
        T[4:8] = np.clip(T[4:8], 0.1, 2.0)
        clean = model_np(T[None], th)[0]
        for nt in N_TR:
            for _ in range(REP):
                rows.append((ci, pt, nt, th, clean + rng.normal(0, sd / np.sqrt(nt), len(th))))
print(f'synthetic: {len(rows)} curves', flush=True)
P0s = np.array([est(r[3], r[4]) for r in rows])
Pf, rs = fit_many([r[3] for r in rows], [r[4] for r in rows], P0s)
np.savez(OUT + '_recovery.npz', cell=[r[0] for r in rows], p_true=[r[1] for r in rows], n_trials=[r[2] for r in rows],
         p_fit=clipped_p(Pf), p_start=clipped_p(P0s), secs=time.time() - t0, n_fits=len(rows),
         n_deadline=sum(r.status == 'deadline' for r in rs))
print(f'synthetic recovery: {len(rows)} fits in {time.time() - t0:.0f} s', flush=True)
