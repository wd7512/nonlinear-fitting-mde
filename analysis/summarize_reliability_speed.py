"""Statistics and figure for reliability_and_speed.py.   usage: python ... IN.json FIG.png"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1])); fig_path = sys.argv[2]
a = lambda x: np.asarray(x, float)
base = d['baseline']; N = d['n']
A_, S_ = a(base['adam_train']), a(base['sym_train']); S100 = a(base['sym100_train'])
print(f'{N} neurons, Adam {d["steps"]} steps (no early stopping), symbfitrs to convergence (1 s budget) and at a 100 ms budget\n')
print('(A) OBJECTIVE QUALITY at the end of the fit (train MSE, all bins of the train repeat), notebook settings')
rat = A_ / S_
print(f'  Adam / symbfitrs train MSE: median {np.median(rat):.4f}; symbfitrs lower by >0.1% in {np.mean(rat > 1.001):.0%}, >1% in {np.mean(rat > 1.01):.0%}, >5% in {np.mean(rat > 1.05):.0%};'
      f' Adam lower by >0.1% in {np.mean(rat < 0.999):.0%}, >1% in {np.mean(rat < 0.99):.0%}')
print(f'  symbfitrs at a 100 ms budget vs at 1 s: median train-MSE ratio {np.median(S100 / S_):.4f}; worse by >1% in {np.mean(S100 / S_ > 1.01):.0%} of neurons; '
      f'held-out R2 median {np.median(base["sym100_r2"]):.3f} vs {np.median(base["sym_r2"]):.3f} at 1 s vs Adam {np.median(base["adam_r2"]):.3f}')
st = np.array(base['sym_status']); print(f'  symbfitrs statuses (1 s): ' + ', '.join(f'{k} {np.mean(st == k):.0%}' for k in sorted(set(st))))
f = a(d['adam_steps_to_match_sym']); m = f > 0
print(f'  Adam reaches symbfitrs\'s final train MSE (within 0.1%) within {d["steps"]} steps in {m.mean():.0%} of neurons; median step {np.median(f[m]) if m.any() else float("nan"):.0f}'
      f' (symbfitrs median {np.median(base["sym_steps"]):.0f} accepted steps, {np.median(base["sym_ms"]):.1f} ms)')
dR = a(base['sym_r2']) - a(base['adam_r2'])
print(f'  held-out R2 (symbfitrs - Adam): median {np.median(dR):+.4f}; start-estimate R2 {np.median(base["start_r2"]):.3f}')

print('\n(B) SENSITIVITY to data scale and learning rate (Adam: lr, 15000 steps, no early stopping; symbfitrs: no tuning parameter)')
print(f'  {"setting":28s} {"Adam test R2":>13s} {"sym test R2":>12s} {"Adam train MSE/sc²":>20s} {"sym train MSE/sc²":>18s} {"Adam within 1% of best":>23s}')
allr = [base] + d['sens']
for r in allr:
    tA, tS = a(r['adam_train']), a(r['sym_train'])
    ref = np.minimum(tA, tS)
    print(f'  {r["label"]:28s} {np.median(r["adam_r2"]):13.3f} {np.median(r["sym_r2"]):12.3f} {np.median(tA):20.4f} {np.median(tS):18.4f} {np.mean(tA <= ref * 1.01):23.0%}')
spread_a = np.ptp([np.median(r['adam_r2']) for r in allr]); spread_s = np.ptp([np.median(r['sym_r2']) for r in allr])
print(f'  range of median test R2 across the five settings: Adam {spread_a:.3f}, symbfitrs {spread_s:.3f}')

print('\n(C) WALL-CLOCK on this 4-core CPU (torch CPU for Adam; Adam time is 15000 steps x the measured per-step cost, not run end to end)')
print(f'  {"neurons":>8s} {"Adam protocol":>14s} {"Adam fit-only":>14s} {"symbfitrs 1 s":>14s} {"+trace scoring":>15s} {"symbfitrs 100 ms":>17s} {"speed-up (1 s / 100 ms)":>25s} {"fits at deadline":>17s}')
for t in d['timing']:
    adam_fit = 15000 * 2 * t['adam_plain_ms_step'] / 1e3
    sp1 = t['adam_protocol_secs'] / (t['sym_solve_secs'] + t['sym_score_secs']); sp2 = t['adam_protocol_secs'] / (t['sym_solve100_secs'] + t['sym_score_secs'])
    print(f'  {t["n"]:8d} {t["adam_protocol_secs"]:13.0f}s {adam_fit:13.0f}s {t["sym_solve_secs"]:13.1f}s {t["sym_score_secs"]:14.1f}s {t["sym_solve100_secs"]:16.1f}s {sp1:12.0f}x / {sp2:.0f}x {t["sym_deadline"]:11d}/{2 * t["n"]}')
last = d['timing'][-1]
print(f'  per-neuron cost at n={last["n"]}: Adam {last["adam_protocol_secs"] / last["n"]:.2f} s, symbfitrs {(last["sym_solve_secs"] + last["sym_score_secs"]) / last["n"] * 1e3:.0f} ms '
      f'({last["sym_solve100_secs"] / last["n"] * 1e3:.0f} ms solver-only at 100 ms); median single fit {last["sym_median_ms"]:.1f} ms')

C1, C2, C3, RD = '#1a5fb4', '#c05a00', '#8a8a8a', '#c01c28'
fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
ax[0, 0].scatter(S_, A_, s=10, alpha=.5, color='#444'); lim = [min(S_.min(), A_.min()) * .9, max(S_.max(), A_.max()) * 1.1]
ax[0, 0].plot(lim, lim, color=RD, ls='--', lw=1); ax[0, 0].set_xscale('log'); ax[0, 0].set_yscale('log'); ax[0, 0].set_xlim(lim); ax[0, 0].set_ylim(lim)
ax[0, 0].set_xlabel('symbfitrs final train MSE'); ax[0, 0].set_ylabel(f'Adam final train MSE ({d["steps"]} steps)')
ax[0, 0].set_title(f'Final objective, per neuron (points above the line: symbfitrs lower)\nsymbfitrs lower by >1% in {np.mean(rat > 1.01):.0%}, Adam lower by >1% in {np.mean(rat < .99):.0%}')
labs = [r['label'].replace(', ', '\n') for r in allr]; x = np.arange(len(allr)); w = .38
ax[0, 1].bar(x - w / 2, [np.median(r['adam_r2']) for r in allr], w, color=C2, label='Adam'); ax[0, 1].bar(x + w / 2, [np.median(r['sym_r2']) for r in allr], w, color=C1, label='symbfitrs')
ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(labs, fontsize=8); lo = min(min(np.median(r['adam_r2']), np.median(r['sym_r2'])) for r in allr)
ax[0, 1].set_ylim(max(0, lo - .05), None); ax[0, 1].set_ylabel('median held-out R²'); ax[0, 1].legend(); ax[0, 1].set_title('Sensitivity to data scale and learning rate')
ns = [t['n'] for t in d['timing']]
ax[1, 0].plot(ns, [t['adam_protocol_secs'] for t in d['timing']], marker='o', color=C2, label='Adam, full protocol (extrapolated from 200 steps)')
ax[1, 0].plot(ns, [2 * 15000 * t['adam_plain_ms_step'] / 1e3 for t in d['timing']], marker='o', color=C2, ls='--', alpha=.6, label='Adam, fit only')
ax[1, 0].plot(ns, [t['sym_solve_secs'] + t['sym_score_secs'] for t in d['timing']], marker='o', color=C1, label='symbfitrs, 1 s budget (incl. trace scoring)')
ax[1, 0].plot(ns, [t['sym_solve100_secs'] + t['sym_score_secs'] for t in d['timing']], marker='o', color=C1, ls='--', label='symbfitrs, 100 ms budget')
ax[1, 0].set_xscale('log'); ax[1, 0].set_yscale('log'); ax[1, 0].set_xlabel('neurons fitted'); ax[1, 0].set_ylabel('wall-clock seconds, 4-core CPU'); ax[1, 0].legend(fontsize=8)
ax[1, 0].set_title('Wall-clock vs number of neurons')
if m.any():
    xs = np.sort(f[m]); ax[1, 1].plot(xs, np.arange(1, len(xs) + 1) / len(f), color=C2, lw=2, label='Adam steps needed'); ax[1, 1].set_xscale('log')
ax[1, 1].axvline(np.median(base['sym_steps']), color=C1, ls='--', label=f'symbfitrs median {np.median(base["sym_steps"]):.0f} accepted steps')
ax[1, 1].set_xlabel('step at which Adam first matches symbfitrs\'s final train MSE'); ax[1, 1].set_ylabel('fraction of neurons (cumulative; plateau < 1 = never matched)')
ax[1, 1].set_title('How long Adam needs to reach the same objective'); ax[1, 1].legend(fontsize=8)
for x_ in ax.flat: x_.spines[['top', 'right']].set_visible(False)
fig.suptitle(f'Objective quality, robustness and CPU speed: Adam vs symbfitrs, {N} Stringer neurons', fontsize=12)
plt.tight_layout(); plt.savefig(fig_path, dpi=140); print('saved', fig_path)
