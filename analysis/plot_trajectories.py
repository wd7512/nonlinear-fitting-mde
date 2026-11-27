"""Learning-trajectory figures from analysis/trajectories.npz (see analysis/trajectories.py)."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

d = np.load('analysis/trajectories.npz', allow_pickle=True)
labels = [str(x) for x in d['labels']]; sgrid, tgrid, best, ab = d['sgrid'], d['tgrid'], d['best'], d['ab']
S, T = d['steps'].astype(float), d['time'].astype(float)            # (methods, neurons, grid)
idx = d['gap_idx']; adam_l, adam_d = d['adam_loss_gap'], d['adam_dead_gap']
LM = labels[:4]; ADAM = labels[4]
COL = {LM[0]: '#8a8a8a', LM[1]: '#1a5fb4', LM[2]: '#26a269', LM[3]: '#6a4ce2', ADAM: '#c05a00'}
NAME = {LM[0]: 'baseline (current solver)', LM[1]: 'step cap 0.1', LM[2]: 'bounds replace clips', LM[3]: 'bounds + step cap 0.3', ADAM: 'Adam, lr 1e-3'}
STYLE = {LM[0]: '-', LM[1]: '-', LM[2]: '-', LM[3]: '-', ADAM: '-'}

# ---------------------------------------------------------------- figure 1: single neurons
final = lambda a: np.array([r[~np.isnan(r)][-1] for r in a])
gap = final(d[f'loss::{LM[0]}']) / adam_l[-1] - 1                                  # LM baseline above Adam, per gap neuron
order = np.argsort(gap); pick = [order[-1], order[len(order) // 2], order[0]]
tags = ['largest gap', 'median gap', 'smallest gap']
fig, ax = plt.subplots(3, 2, figsize=(14, 11.5))
for r, (g, tag) in enumerate(zip(pick, tags)):
    n = idx[g]
    for k in LM:
        y = d[f'loss::{k}'][g]; y = y[~np.isnan(y)]; x = np.arange(len(y))
        ax[r, 0].plot(x, y, color=COL[k], marker='o', ms=2.5, lw=1.5, label=NAME[k])
        z = d[f'dead::{k}'][g]; z = z[~np.isnan(z)]; ax[r, 1].plot(np.arange(len(z)), z, color=COL[k], lw=1.8, drawstyle='steps-post')
    ax[r, 0].plot(np.arange(len(adam_l)), adam_l[:, g], color=COL[ADAM], lw=1.5, label=NAME[ADAM])
    ax[r, 1].plot(np.arange(len(adam_d)), adam_d[:, g], color=COL[ADAM], lw=1.8, drawstyle='steps-post')
    ax[r, 0].axhline(best[n], color='k', ls=':', lw=.8, label='best result any method found')
    for a in ax[r]: a.set_xscale('symlog', linthresh=1); a.set_xlim(0, 15000)
    ax[r, 0].set_yscale('log'); ax[r, 0].set_ylabel('train MSE')
    ax[r, 0].set_title(f'neuron row {int(d["rows"][n])} ({tag}: baseline LM ends {100 * gap[g]:.1f}% above Adam)', fontsize=10)
    ax[r, 1].set_ylabel('parameters in a dead zone (of 9)'); ax[r, 1].set_ylim(-.2, 9.2)
    if r == 0: ax[r, 0].legend(fontsize=8)
    if r == 2: [a.set_xlabel('step (LM: accepted steps; Adam: updates)') for a in ax[r]]
ax[0, 1].set_title('Dead-zone parameters along the same trajectories', fontsize=10)
for a in ax.flat: a.spines[['top', 'right']].set_visible(False)
fig.suptitle('Learning trajectories on single neurons: baseline LM, proposed fixes and Adam', fontsize=12); plt.tight_layout(); plt.savefig('analysis/trajectories_neurons.png', dpi=140)

# ---------------------------------------------------------------- figure 2: all 300 neurons
exc = lambda A: A / best[None, :, None] - 1
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
for m, k in enumerate(labels):
    e = exc(S)[m]; med = np.median(e, 0); q1, q3 = np.percentile(e, [25, 75], 0)
    ax[0].plot(sgrid, np.maximum(100 * med, 1e-6), color=COL[k], lw=2, label=NAME[k]); ax[0].fill_between(sgrid, np.maximum(100 * q1, 1e-6), np.maximum(100 * q3, 1e-6), color=COL[k], alpha=.10)
    ax[1].plot(sgrid, 100 * np.mean(e <= 1e-3, 0), color=COL[k], lw=2, label=f'{NAME[k]}  ({100 * np.mean(e[:, -1] <= 1e-3):.0f}% at the end)')
    et = (T[m] / best[:, None] - 1); ax[2].plot(tgrid, 100 * np.mean(et <= 1e-3, 0), color=COL[k], lw=2, label=NAME[k])
ax[0].set_yscale('log'); ax[0].set_ylabel('excess train MSE over the best result (%): median, IQR band'); ax[0].set_title('How far each method is from the best result')
ax[1].set_ylabel('% of 300 neurons within 0.1% of the best result'); ax[1].set_title('Reaching the target, by step'); ax[1].legend(fontsize=8, loc='upper left')
ax[2].set_ylabel('% of 300 neurons within 0.1% of the best result'); ax[2].set_xscale('log'); ax[2].set_xlabel('time per fit (ms, log scale)')
amort = float(d['adam_amortised_ms_per_step']) * 1e3
ax[2].set_title(f'Reaching the target, by time (Adam: batched run, {amort:.0f} µs per step per neuron amortised)', fontsize=9); ax[2].legend(fontsize=8, loc='upper left')
for a in ax[:2]: a.set_xscale('symlog', linthresh=1); a.set_xlim(0, 15000); a.set_xlabel('step (LM: accepted steps; Adam: updates)')
ax[1].set_ylim(0, 100); ax[2].set_ylim(0, 100)
for a in ax: a.spines[['top', 'right']].set_visible(False)
fig.suptitle('Aggregate learning trajectories, 300 Stringer neurons', fontsize=12); plt.tight_layout(); plt.savefig('analysis/trajectories_aggregate.png', dpi=140)
print('ok; plotted neurons', [int(d['rows'][idx[g]]) for g in pick])
