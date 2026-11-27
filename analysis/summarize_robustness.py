"""Summarise robustness_*.json: statistics to stdout, figure to the given path."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

data = json.load(open(sys.argv[1])); fig_path = sys.argv[2]
splits = data['splits']; seeds = list(splits)
a = lambda x: np.asarray(x, float)
rng = np.random.default_rng(0)


def boot_ci(x, n=4000):
    m = [np.median(rng.choice(x, len(x))) for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


N = len(data['rows'])
print(f'{N} random neurons passing SF>0.6, C>0.3; validation-split seeds {seeds}\n')
print('RULES: notebook = val1-best + patience (50..500) then val2; val1_best = no patience; no_stop = run to the end')
for sd in seeds:
    s = splits[sd]; A, S = s['adam'], s['sym']
    print(f'--- split seed {sd} ---')
    print(f"val1-best step    Adam median {np.median(A['best']):.0f} (IQR {np.percentile(A['best'],25):.0f}-{np.percentile(A['best'],75):.0f})"
          f" | symbfitrs median {np.median(S['best']):.0f} (IQR {np.percentile(S['best'],25):.0f}-{np.percentile(S['best'],75):.0f})")
    print(f"notebook stop     Adam median {np.median(A['stop']):.0f} | symbfitrs median {np.median(S['stop']):.0f}"
          f"  (symbfitrs stop == its last step in {np.mean(a(S['stop']) >= a(S['steps']) - 0):.0%} of cells)")
    print(f"validation-1 gain over start (>1%):  Adam {np.mean(a(A['val1_gain'])>0.01):.0%} of cells | symbfitrs {np.mean(a(S['val1_gain'])>0.01):.0%}")
    print(f"train-fit accepted steps: median {np.median(S['steps']):.0f}, p95 {np.percentile(S['steps'],95):.0f}; status {dict(zip(*np.unique(S['status'], return_counts=True)))}")
    print(f"all-angle refit steps:    median {np.median(S['refit_steps']):.0f}, p95 {np.percentile(S['refit_steps'],95):.0f}; "
          f"status {dict(zip(*np.unique(S['refit_status'], return_counts=True)))}; median {np.median(S['refit_ms']):.1f} ms (Adam {A['secs']/N*1000:.0f} ms/cell amortised)")
    print(f"median test R2: start {np.median(s['start_r2']):.3f}")
    for k in ('notebook', 'val1_best', 'no_stop'):
        ra, rs = a(A['r2'][k]), a(S['r2'][k]); d = rs - ra; lo, hi = boot_ci(d)
        print(f"  {k:10s} Adam {np.median(ra):.3f}  symbfitrs {np.median(rs):.3f}  median paired diff (sym-Adam) {np.median(d):+.4f}"
              f" [95% CI {lo:+.4f}, {hi:+.4f}]  sym>Adam in {np.mean(d>0):.0%}  Wilcoxon p={wilcoxon(rs, ra).pvalue:.2g}")
    print(f"  prefix property (capped == uncapped prefix) verified on {s['prefix_checked']} cells\n")
if len(seeds) > 1:
    for k in ('notebook',):
        dd = [a(splits[sd]['sym']['r2'][k]) - a(splits[sd]['adam']['r2'][k]) for sd in seeds]
        print(f"cross-split stability, per-cell ΔR² (sym-Adam, {k}): median by split {[round(float(np.median(d)),4) for d in dd]};"
              f" pairwise corr {[round(float(np.corrcoef(dd[i], dd[j])[0,1]),2) for i in range(len(dd)) for j in range(i+1,len(dd))]}")

s0 = splits[seeds[0]]; A, S = s0['adam'], s0['sym']
cA, cS = '#c05a00', '#1a5fb4'
fig, ax = plt.subplots(2, 2, figsize=(12.5, 9))
# (a) where validation loss bottoms out
for lab, x, c in (('Adam', A['best'], cA), ('symbfitrs', S['best'], cS)):
    for sd in seeds:
        xs = np.sort(np.maximum(splits[sd]['adam' if lab == 'Adam' else 'sym']['best'], 0.5))
        ax[0, 0].plot(xs, np.arange(1, N + 1) / N, color=c, alpha=1 if sd == seeds[0] else .35, lw=2 if sd == seeds[0] else 1,
                      label=lab if sd == seeds[0] else None)
ax[0, 0].set_xscale('log'); ax[0, 0].set_xlabel('step where validation-1 MSE is lowest (0 plotted at 0.5)')
ax[0, 0].set_ylabel('fraction of neurons'); ax[0, 0].set_title('Where validation loss bottoms out\n(faint lines: other validation splits)'); ax[0, 0].legend()
# (b) paired test R2, notebook rule
ra, rs = a(A['r2']['notebook']), a(S['r2']['notebook'])
ax[0, 1].scatter(ra, rs, s=12, alpha=.6, color='#444'); lim = [min(ra.min(), rs.min()) - .02, 1]
ax[0, 1].plot(lim, lim, 'r--', lw=1); ax[0, 1].set_xlim(lim); ax[0, 1].set_ylim(lim)
ax[0, 1].set_xlabel('Adam test R² (notebook stopping rule)'); ax[0, 1].set_ylabel('symbfitrs test R²')
ax[0, 1].set_title(f'Held-out R², one point per neuron (median Δ {np.median(rs-ra):+.4f})')
# (c) paired difference by rule
rules = ('notebook', 'val1_best', 'no_stop'); labels = ('notebook\nrule', 'val1-best\nonly', 'no stopping\n(run to end)')
vals = [a(S['r2'][k]) - a(A['r2'][k]) for k in rules]
ax[1, 0].boxplot(vals, showfliers=False, widths=.5, tick_labels=labels); ax[1, 0].axhline(0, color='r', ls='--', lw=1)
for i, v in enumerate(vals, 1): ax[1, 0].scatter(rng.normal(i, .06, len(v)), v, s=6, alpha=.35, color='#444')
ax[1, 0].set_ylabel('test R², symbfitrs − Adam'); ax[1, 0].set_title('Does the conclusion depend on the stopping rule?')
ax[1, 0].set_ylim(-0.1, 0.1)
# (d) accepted steps symbfitrs takes
ax[1, 1].hist([np.clip(S['steps'], 1, None), np.clip(S['refit_steps'], 1, None)], bins=np.logspace(0, 4.2, 30),
              label=['train fit', 'all-angle refit'], color=['#7aa6d8', cS])
ax[1, 1].set_xscale('log'); ax[1, 1].set_xlabel('accepted steps until the solver stops'); ax[1, 1].set_ylabel('neurons')
dl = sum(x == 'deadline' for x in S['refit_status'])
ax[1, 1].set_title(f"symbfitrs steps per fit ({dl} of {N} refits hit the 1 s budget)"); ax[1, 1].legend()
for x in ax.flat: x.spines[['top', 'right']].set_visible(False)
fig.suptitle(f'{N} random Stringer-2019 neurons, generalised-Gaussian model, jacob_tuning validation/refit protocol', fontsize=12)
plt.tight_layout(); plt.savefig(fig_path, dpi=140); print('saved', fig_path)
