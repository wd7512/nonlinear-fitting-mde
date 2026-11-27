"""Statistics and figure for exponent_identifiability.py output.   usage: python ... PREFIX FIG.png"""
import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

pre, fig_path = sys.argv[1], sys.argv[2]
b = np.load(pre + '_boot.npz'); rc = np.load(pre + '_recovery.npz')
rng = np.random.default_rng(0)
N, B = b['boot_p'].shape[:2]
pmin_hat = b['p_hat'].min(1)
double = b['amp_hat'].min(1) / np.maximum(b['amp_hat'].max(1), 1e-12) >= 0.1
pmin_boot = b['boot_p'].min(2)                                   # (cells, B)
amp_ok = b['boot_amp'].min(2) / np.maximum(b['boot_amp'].max(2), 1e-12) >= 0.1
lo, hi = np.percentile(pmin_boot, [2.5, 97.5], axis=1); sd = pmin_boot.std(1)
dd = double
print(f'{N} neurons, B={B} trial-level bootstraps ({b["n_fits"]} fits in {float(b["secs"]):.0f} s wall on 4 threads; '
      f'{int(b["n_deadline"])} = {int(b["n_deadline"]) / int(b["n_fits"]):.0%} hit the 100 ms budget)')
print(f'double-peaked (weaker/stronger amplitude >= 0.1): {dd.sum()} of {N}')
print('\n(1) real neurons, double-peaked only')
print(f'  min(p) point estimate: median {np.median(pmin_hat[dd]):.2f}, IQR {np.percentile(pmin_hat[dd],25):.2f}-{np.percentile(pmin_hat[dd],75):.2f}')
print(f'  bootstrap SD of min(p): median {np.median(sd[dd]):.2f}, IQR {np.percentile(sd[dd],25):.2f}-{np.percentile(sd[dd],75):.2f}')
w = (hi - lo)[dd]
print(f'  95% interval width: median {np.median(w):.2f}; width <= 1.0 (the notebook\'s +/-0.5 tolerance) in {np.mean(w <= 1):.0%} of neurons; > 2 in {np.mean(w > 2):.0%}')
print(f'  interval excludes p = 2 (Gaussian): {np.mean((hi[dd] < 2) | (lo[dd] > 2)):.0%} of neurons ({np.mean(hi[dd] < 2):.0%} peakier, {np.mean(lo[dd] > 2):.0%} flatter)')
at_bound = np.mean((pmin_boot[dd] <= 0.1001) | (pmin_boot[dd] >= 3.999))
print(f'  bootstrap fits with min(p) pinned at a bound (0.1 or 4): {at_bound:.0%}')
rel = 1 - np.mean(sd[dd] ** 2) / np.var(pmin_hat[dd])
print(f'  between-neuron variance of min(p) explained by sampling noise: {1 - rel:.0%}  (reliability {rel:.2f})')
# split-half, pairing peaks by centre as the notebook does
th_e, th_o = b['theta_hat'], b['theta_odd']; swap = np.abs(np.arctan2(np.sin(th_e - th_o), np.cos(th_e - th_o))) > np.pi / 2
p_o = np.where(swap[:, None], b['p_odd'][:, ::-1], b['p_odd']); a_o = np.where(swap[:, None], b['amp_odd'][:, ::-1], b['amp_odd'])
dbl = dd & (a_o.min(1) / np.maximum(a_o.max(1), 1e-12) >= 0.1)
repro = dbl & (np.abs(b['p_hat'] - p_o) <= 0.5).all(1)
print(f'  split-half (even vs odd fits): double in both {dbl.sum()}, exponents agree within 0.5: {repro.sum()} ({repro.sum() / N:.0%} of all, {repro.sum() / max(dbl.sum(),1):.0%} of double)')
print(f'  cells passing the notebook reproducibility filter have median bootstrap SD {np.median(sd[repro]):.2f} vs {np.median(sd[dbl & ~repro]):.2f} for the rest')
pop_med = [np.median(pmin_hat[dd][rng.integers(0, dd.sum(), dd.sum())]) for _ in range(2000)]
print(f'  population median of min(p): {np.median(pmin_hat[dd]):.2f} (cell-bootstrap 95% CI {np.percentile(pop_med,2.5):.2f}-{np.percentile(pop_med,97.5):.2f});'
      f' median of per-cell bootstrap means {np.median(pmin_boot[dd].mean(1)):.2f}; reproducible-only median {np.median(((b["p_hat"] + p_o) / 2).min(1)[repro]):.2f}')

print('\n(2) synthetic recovery (known p, noise matched to each neuron, both peaks real)')
pt, nt, pf, ps = rc['p_true'], rc['n_trials'], rc['p_fit'], rc['p_start']
err = lambda p: (p - pt[:, None])
for n in (4, 8, 16, 32, 64):
    m = nt == n
    ef, es = err(pf)[m], err(ps)[m]
    print(f'  {n:2d} trials/bin: RMSE fit {np.sqrt((ef**2).mean()):.2f} vs estimator {np.sqrt((es**2).mean()):.2f};  within +/-0.5: fit {np.mean(np.abs(ef) <= .5):.0%}, estimator {np.mean(np.abs(es) <= .5):.0%};'
          f'  bias fit {ef.mean():+.2f}')
print('  at 4 trials/bin (the real training half), by true p:')
for p in sorted(set(pt)):
    m = (nt == 4) & (pt == p)
    print(f'    p={p}: fit median {np.median(pf[m]):.2f} (IQR {np.percentile(pf[m],25):.2f}-{np.percentile(pf[m],75):.2f}), '
          f'estimator median {np.median(ps[m]):.2f}; within +/-0.5: fit {np.mean(np.abs(err(pf)[m]) <= .5):.0%}, estimator {np.mean(np.abs(err(ps)[m]) <= .5):.0%}')
print(f'  {int(rc["n_fits"])} synthetic fits in {float(rc["secs"]):.0f} s wall')

# ------------------------------------------------------------------ figure
C_FIT, C_EST, C_TRUE = '#1a5fb4', '#8a8a8a', '#c01c28'
fig, ax = plt.subplots(2, 2, figsize=(12.5, 9))
idx = np.where(dd)[0]; idx = idx[np.argsort(pmin_hat[idx])]; idx = idx[np.linspace(0, len(idx) - 1, 70).astype(int)]
for j, i in enumerate(idx):
    ax[0, 0].plot([lo[i], hi[i]], [j, j], color=C_FIT, alpha=.45, lw=2); ax[0, 0].scatter(pmin_hat[i], j, s=10, color='k', zorder=3)
ax[0, 0].axvline(2, color=C_TRUE, ls='--', lw=1, label='p = 2 (Gaussian)'); ax[0, 0].set_ylabel('70 neurons, sorted by point estimate')
ax[0, 0].set_xlabel('min(p1, p2)'); ax[0, 0].set_yticks([]); ax[0, 0].legend(loc='lower right')
ax[0, 0].set_title('Per-neuron exponent with 95% bootstrap interval\n(dot: point estimate; bar: trial-resampling interval)')
ax[0, 1].hist(np.clip(sd[dd], 0, 1.6), bins=30, color=C_FIT, alpha=.85); ax[0, 1].axvline(0.25, color=C_TRUE, ls='--', lw=1)
ax[0, 1].set_xlabel('bootstrap SD of min(p) (clipped at 1.6)'); ax[0, 1].set_ylabel('double-peaked neurons')
ax[0, 1].set_title(f'Sampling noise in the exponent (median SD {np.median(sd[dd]):.2f})\nred: SD 0.25 = the notebook\'s ±0.5 at 2 SD')
ns = np.array([4, 8, 16, 32, 64])
for lab, p, c in (('symbfitrs fit', pf, C_FIT), ('estimator only (start)', ps, C_EST)):
    ax[1, 0].plot(ns, [np.sqrt((err(p)[nt == n] ** 2).mean()) for n in ns], marker='o', color=c, label=lab)
ax[1, 0].axvline(4, color='k', ls=':', lw=1); ax[1, 0].annotate('real training\nhalf (4/bin)', (4, ax[1, 0].get_ylim()[1]), xytext=(5, -28), textcoords='offset points', fontsize=8)
ax[1, 0].set_xscale('log', base=2); ax[1, 0].set_xticks(ns); ax[1, 0].set_xticklabels(ns); ax[1, 0].set_xlabel('trials per angle bin')
ax[1, 0].set_ylabel('RMSE of recovered exponent'); ax[1, 0].set_title('Synthetic recovery: error vs amount of data'); ax[1, 0].legend()
for n, ls in ((4, '-'), (32, '--')):
    ps_ = sorted(set(pt)); med = [np.median(pf[(nt == n) & (pt == p)]) for p in ps_]
    q1 = [np.percentile(pf[(nt == n) & (pt == p)], 25) for p in ps_]; q3 = [np.percentile(pf[(nt == n) & (pt == p)], 75) for p in ps_]
    ax[1, 1].plot(ps_, med, marker='o', color=C_FIT, ls=ls, label=f'fit, {n}/bin (median, IQR)'); ax[1, 1].fill_between(ps_, q1, q3, color=C_FIT, alpha=.12 if n == 4 else .06)
ax[1, 1].plot(ps_, [np.median(ps[(nt == 4) & (pt == p)]) for p in ps_], marker='s', color=C_EST, label='estimator only, 4/bin')
ax[1, 1].plot([0, 3.2], [0, 3.2], color=C_TRUE, ls='--', lw=1, label='truth'); ax[1, 1].set_xlabel('true exponent'); ax[1, 1].set_ylabel('recovered exponent')
ax[1, 1].set_title('Synthetic recovery: bias and spread'); ax[1, 1].legend(fontsize=8)
for x in ax.flat: x.spines[['top', 'right']].set_visible(False)
fig.suptitle(f'How well is the peakiness exponent identified? {N} Stringer neurons, {B} bootstraps/neuron, {int(rc["n_fits"]):,} synthetic fits', fontsize=12)
plt.tight_layout(); plt.savefig(fig_path, dpi=140); print('saved', fig_path)
