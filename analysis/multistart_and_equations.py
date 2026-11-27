"""(A) Multi-start: does the notebook's single estimator start find the best optimum?
(B) Model discovery: held-out R^2 of the benchmark's ten candidate tuning equations vs the
notebook's Gaussian / generalised Gaussian, all fitted with symbfitrs on the same neurons.

usage: python analysis/multistart_and_equations.py OUT_PREFIX
"""
import sys, time
import numpy as np
from _neuro import *
from bench.suites.stringer2019 import EQUATIONS, estimate_peaks, initial_params

OUT = sys.argv[1]
rng = np.random.default_rng(2)
est = lambda th, y: np.asarray(estimator(th, y), dtype=np.float64)
cells = [trials(r) for r in ROWS]
TH = [c[0] for c in cells]; TRAIN = [c[1][0].mean(1) for c in cells]; TEST = [c[1][1].mean(1) for c in cells]

# ---------------------------------------------------------------- (A) multi-start, notebook model
K = 16; t0 = time.time()
P0 = np.array([est(TH[i], TRAIN[i]) for i in range(len(ROWS))])
starts = np.repeat(P0[:, None], K, 1)
for k in range(1, K):
    s = starts[:, k]
    s[:, 0] += rng.normal(0, 0.15, len(s))                                   # preferred direction
    s[:, 1:4] = s[:, 1:4] * np.exp(rng.normal(0, 0.3, (len(s), 3))) + rng.uniform(0, 0.05, (len(s), 3))
    s[:, 4:8] = np.clip(s[:, 4:8], 0.1, 2) * np.exp(rng.normal(0, 0.4, (len(s), 4)))
    s[:, 8:10] = rng.uniform(0.5, 3.0, (len(s), 2))                          # exponents anywhere plausible
flat = [(i, k) for i in range(len(ROWS)) for k in range(K)]
Pf, rs = fit_many([TH[i] for i, _ in flat], [TRAIN[i] for i, _ in flat], np.array([starts[i, k] for i, k in flat]))
loss = np.array([r.loss for r in rs]).reshape(len(ROWS), K)
test_r2 = np.array([r2(TEST[i], model_np(Pf[j:j + 1], TH[i])[0]) for j, (i, k) in enumerate(flat)]).reshape(len(ROWS), K)
pfit = clipped_p(Pf).reshape(len(ROWS), K, 2)
np.savez(OUT + '_multistart.npz', loss=loss, test_r2=test_r2, p_fit=pfit, secs=time.time() - t0, n_fits=len(flat),
         status=np.array([r.status for r in rs]).reshape(len(ROWS), K))
print(f'multi-start: {len(flat)} fits in {time.time() - t0:.0f} s', flush=True)

# ---------------------------------------------------------------- (B) candidate equations
t0 = time.time()
names_eq = list(EQUATIONS)
compiled = {e: symbfitrs.compile(EQUATIONS[e][1].replace('Abs(', 'abs('), variables=['theta']) for e in names_eq}
compiled['gen_gauss_nb'] = MODEL
KS = 4                                                                        # estimate + 3 perturbed starts
rec = {e: dict(test_r2=np.full(len(ROWS), np.nan), train_loss=np.full(len(ROWS), np.nan), n_params=0, ms=0.0, status=[]) for e in compiled}
for e, m in compiled.items():
    jobs = []
    for i in range(len(ROWS)):
        if e == 'gen_gauss_nb':
            base = est(TH[i], TRAIN[i])[ORDER]
        else:
            base_dict = dict(zip(EQUATIONS[e][0], initial_params(e, estimate_peaks(TH[i], TRAIN[i]))))
            base = np.array([base_dict[p] for p in m.params])
        for k in range(KS):
            p0 = base if k == 0 else base + 0.3 * np.maximum(np.abs(base), 0.1) * rng.standard_normal(len(base))
            jobs.append((m, TH[i], TRAIN[i], p0))
    tt = time.time()
    res = symbfitrs.Pool().fit_many(jobs, budget_ms=100.0, progress=False)
    rec[e]['ms'] = (time.time() - tt) * 1e3 / len(jobs); rec[e]['n_params'] = len(m.params)
    for i in range(len(ROWS)):
        rr = res[i * KS:(i + 1) * KS]
        ok = [r for r in rr if np.isfinite(r.loss)]
        if not ok: continue
        best = min(ok, key=lambda r: r.loss)
        rec[e]['train_loss'][i] = best.loss
        rec[e]['test_r2'][i] = r2(TEST[i], m.predict(TH[i], best.params))
        rec[e]['status'].append(best.status)
    print(f'  {e:16s} {len(m.params):2d} params, {rec[e]["ms"]:.1f} ms/fit wall (4 threads), median test R2 {np.nanmedian(rec[e]["test_r2"]):.3f}', flush=True)
np.savez(OUT + '_equations.npz', names=list(compiled), n_params=[rec[e]['n_params'] for e in compiled],
         test_r2=np.array([rec[e]['test_r2'] for e in compiled]), train_loss=np.array([rec[e]['train_loss'] for e in compiled]),
         ms=[rec[e]['ms'] for e in compiled], secs=time.time() - t0, n_fits=len(compiled) * len(ROWS) * KS)
print(f'equations: {len(compiled) * len(ROWS) * KS} fits in {time.time() - t0:.0f} s')
