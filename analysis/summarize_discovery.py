"""Statistics and figure for multistart_and_equations.py output.   usage: python ... PREFIX FIG.png"""
import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

pre, fig_path = sys.argv[1], sys.argv[2]
ms = np.load(pre + '_multistart.npz'); eq = np.load(pre + '_equations.npz', allow_pickle=True)
N, K = ms['loss'].shape
loss, tr2, pf = ms['loss'], ms['test_r2'], ms['p_fit']
best = np.nanmin(loss, 1); gain = (loss[:, 0] - best) / loss[:, 0]
print(f'(A) multi-start: {N} neurons x {K} starts = {int(ms["n_fits"])} fits in {float(ms["secs"]):.0f} s wall (4 threads)')
print(f'  the estimator start (k=0) reaches the best train loss (within 0.1%) in {np.mean(gain < 1e-3):.0%} of neurons')
print(f'  a better optimum exists: >1% lower train MSE in {np.mean(gain > .01):.0%}, >10% in {np.mean(gain > .10):.0%}; median gain among those {np.median(gain[gain > .01]):.1%}')
kbest = np.nanargmin(loss, 1); d = tr2[np.arange(N), kbest] - tr2[:, 0]
print(f'  test R2 of the best-train start vs the estimator start: median paired diff {np.median(d):+.4f}, mean {d.mean():+.4f}; '
      f'improves in {np.mean(d > 1e-3):.0%}, worsens in {np.mean(d < -1e-3):.0%} (Wilcoxon p={wilcoxon(d).pvalue:.2g})')
near = loss <= best[:, None] * 1.01                                   # starts within 1% of the best train loss
pm = pf.min(2); rng_p = np.array([np.ptp(pm[i][near[i]]) for i in range(N)])
nn = near.sum(1)
print(f'  near-optimal starts per neuron (within 1% of best train MSE): median {np.median(nn):.0f} of {K}')
print(f'  spread of min(p) among those near-optimal solutions: median {np.median(rng_p):.2f}; > 0.5 in {np.mean(rng_p > .5):.0%}; > 1.0 in {np.mean(rng_p > 1):.0%} of neurons')
print(f'  starts ending at the 100 ms deadline: {np.mean(ms["status"] == "deadline"):.0%}')

names = [str(n) for n in eq['names']]; R = eq['test_r2']; L = eq['train_loss']; npar = eq['n_params']; ref = names.index('gen_gauss_nb')
print(f'\n(B) {len(names)} candidate equations x {N} neurons x 4 starts = {int(eq["n_fits"])} fits in {float(eq["secs"]):.0f} s wall')
rank = (-np.nan_to_num(R, nan=-9)).argsort(0).argsort(0) + 1
win = np.bincount(np.nanargmax(np.where(np.isnan(R), -9, R), 0), minlength=len(names)) / N
print(f'  {"equation":16s}{"params":>7s}{"median test R2":>16s}{"vs gen_gauss_nb":>26s}{"mean rank":>11s}{"best in":>9s}{"ms/fit":>8s}')
for j in np.argsort(-np.nanmedian(R, 1)):
    dj = R[j] - R[ref]; ok = np.isfinite(dj)
    ci = ''
    if j != ref: ci = f'{np.median(dj[ok]):+.4f} (better in {np.mean(dj[ok] > 0):.0%})'
    print(f'  {names[j]:16s}{int(npar[j]):7d}{np.nanmedian(R[j]):16.3f}{ci:>26s}{rank[j].mean():11.1f}{win[j]:9.0%}{eq["ms"][j]:8.1f}')
# honest per-neuron model selection: BIC on the training bins only (214 bins), test R2 of the selected model
n_bins = 214
bic = n_bins * np.log(np.maximum(L, 1e-12)) + npar[:, None] * np.log(n_bins)
sel = np.nanargmin(np.where(np.isnan(bic), np.inf, bic), 0); r_sel = R[sel, np.arange(N)]
print(f'\n  per-neuron selection by train-only BIC: median test R2 {np.median(r_sel):.3f} (gen_gauss_nb {np.nanmedian(R[ref]):.3f}, best single equation '
      f'{max(np.nanmedian(R, 1)):.3f}); selected: ' + ', '.join(f'{names[j]} {np.mean(sel == j):.0%}' for j in np.argsort(-np.bincount(sel, minlength=len(names)))[:5]))
dsel = r_sel - R[ref]
print(f'  BIC-selected minus gen_gauss_nb: median {np.median(dsel):+.4f}; better in {np.mean(dsel > 0):.0%}; Wilcoxon p={wilcoxon(dsel[np.isfinite(dsel)]).pvalue:.2g}')

C = '#1a5fb4'; G = '#8a8a8a'; Rd = '#c01c28'
fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
o = np.argsort(np.nanmedian(R, 1))
ax[0, 0].barh([names[j] for j in o], [np.nanmedian(R[j]) for j in o], color=[Rd if names[j] == 'gen_gauss_nb' else C for j in o], alpha=.85)
for i, j in enumerate(o): ax[0, 0].text(np.nanmedian(R[j]) + .002, i, f'{int(npar[j])} params', va='center', fontsize=8)
lo_ = np.nanmin([np.nanmedian(R[j]) for j in range(len(names))]); ax[0, 0].set_xlim(lo_ - .03, np.nanmax([np.nanmedian(R[j]) for j in range(len(names))]) + .05)
ax[0, 0].set_xlabel('median held-out R² (odd repeat)'); ax[0, 0].set_title('Candidate tuning equations, 300 neurons\n(red: notebook generalised Gaussian)')
dd = [R[j] - R[ref] for j in o if j != ref]
bp = ax[0, 1].boxplot(dd, vert=False, showfliers=False, tick_labels=[names[j] for j in o if j != ref]); ax[0, 1].axvline(0, color=Rd, ls='--', lw=1)
ax[0, 1].set_xlabel('held-out R² minus generalised Gaussian (per neuron)'); ax[0, 1].set_title('Paired differences to the notebook model')
ax[1, 0].hist(np.clip(gain * 100, 0, 30), bins=np.linspace(0, 30, 31), color=C, alpha=.85); ax[1, 0].set_yscale('log')
ax[1, 0].set_xlabel('% lower train MSE of the best of 16 starts vs the estimator start (clipped at 30)'); ax[1, 0].set_ylabel('neurons (log)')
ax[1, 0].set_title(f'Multi-start: does the estimator start find the optimum?\n{np.mean(gain > .01):.0%} of neurons have a >1% better optimum')
xs = np.sort(rng_p); ax[1, 1].plot(xs, np.arange(1, N + 1) / N, color=C, lw=2); ax[1, 1].axvline(.5, color=Rd, ls='--', lw=1)
ax[1, 1].set_xlabel('range of min(p) across starts within 1% of the best train MSE'); ax[1, 1].set_ylabel('fraction of neurons')
ax[1, 1].set_title('How flat is the fit in the exponent?')
for x in ax.flat: x.spines[['top', 'right']].set_visible(False)
fig.suptitle('Multi-start robustness and model discovery with symbfitrs on Stringer 2019 neurons', fontsize=12)
plt.tight_layout(); plt.savefig(fig_path, dpi=140); print('saved', fig_path)
