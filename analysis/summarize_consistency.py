"""Summarise analysis/consistency/*.npz: accuracy, optimum quality and self-consistency per arm,
and draw analysis/consistency.png.  usage: PYTHONPATH=analysis python analysis/summarize_consistency.py"""
import json, os
import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import consistency as _c
from consistency import TH, HALF, CELLS, P0, N, P_MIN, P_MAX, gen_gauss, OUT, SYNTH
TRUTH = _c.TRUTH if SYNTH else None
PNG = 'analysis/consistency_synth' if SYNTH else 'analysis/consistency'

LABEL = {'A': 'A  Adam notebook', 'B': 'B  our notebook (penalised, λ=1)', 'C': 'C  symbfitrs today (clipped)',
         'D': 'D  bounds', 'E': 'E  bounds + 8 starts', 'F': 'F  bounds + 8 starts + CV penalty'}
COLOR = dict(zip('ABCDEF', ['#257de0', '#f47230', '#00b894', '#f2a600', '#ee7fac', '#008a17']))
arms = [a for a in 'ABCDEF' if os.path.exists(f'{OUT}/{a}.npz')]
R = {a: dict(np.load(f'{OUT}/{a}.npz')) for a in arms}
boots = {a: np.load(f'{OUT}/boot_{a}.npz')['P'] for a in 'DEF' if os.path.exists(f'{OUT}/boot_{a}.npz')}
MIN_AMP_RATIO, P_REPRO_TOL = 0.1, 0.5


def predict(P, theta=None):
    th = TH if theta is None else theta
    with torch.no_grad():
        return gen_gauss(torch.tensor(th, dtype=torch.float64), torch.tensor(np.asarray(P, np.float64))).squeeze(0).numpy()


def mse(P, half):
    return ((predict(P) - HALF[half]) ** 2).mean(1)


def r2(P, half):
    y = HALF[half]
    return 1 - ((y - predict(P)) ** 2).sum(1) / ((y - y.mean(1, keepdims=True)) ** 2).sum(1)


def peaks(P):          # as the notebook: centre, amplitudes and exponents as the model uses them
    P = np.asarray(P)
    return P[:, 0], P[:, 2:4].clip(min=0), P[:, 8:10].clip(P_MIN, P_MAX)


def paired(Pa, Pb):
    """The notebook's pairing: (A_a, p_a), (A_b, p_b) with b's labels swapped onto a's peaks."""
    ta, Aa, pa = peaks(Pa); tb, Ab, pb = peaks(Pb)
    swap = (np.abs(np.arctan2(np.sin(ta - tb), np.cos(ta - tb))) > np.pi / 2)[:, None]
    return Aa, pa, np.where(swap, Ab[:, ::-1], Ab), np.where(swap, pb[:, ::-1], pb)


double = lambda A: A.min(1) / np.maximum(A.max(1), 1e-12) >= MIN_AMP_RATIO


def icc(x, y):         # one-way ICC(1,1) for two measurements per cell
    d = np.stack([x, y], 1); k = 2; gm = d.mean()
    msb = k * ((d.mean(1) - gm) ** 2).sum() / (len(d) - 1)
    msw = ((d - d.mean(1, keepdims=True)) ** 2).sum() / (len(d) * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw)


def boot_ci(x, f=np.median, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])


# ---- the reference optimum: the lowest even-half MSE any arm (or any start/run) reached
cands = [mse(R[a]['even'], 'even') for a in arms] + [mse(P, 'even') for a in arms for P in R[a]['rr']]
best = np.min(cands, 0)
# noise of the even-half mean curve, per bin: SD over its 4 trials / 2; RMS over bins
noise = np.array([np.sqrt((c[1][0].std(1, ddof=1) ** 2 / c[1][0].shape[1]).mean()) for c in CELLS])
fine = np.tile(np.linspace(0, 2 * np.pi, 361), (N, 1))

rows, S = [], {}
r2_A = r2(R['A']['even'], 'odd') if 'A' in R else None
curve_A = predict(R['A']['even'], fine) if 'A' in R else None
for a in arms:
    Pe, Po, rr = R[a]['even'], R[a]['odd'], R[a]['rr']
    s = {}
    s['test_r2'] = r2(Pe, 'odd'); s['test_r2_rev'] = r2(Po, 'even')
    L = mse(Pe, 'even'); s['excess'] = L / best - 1
    Ae, pe, Ao, po = paired(Pe, Po)
    dbl = double(Ae) & double(Ao)
    mpe, mpo = pe.min(1), po.min(1)
    repro = dbl & (np.abs(pe - po) <= P_REPRO_TOL).all(1)
    s.update(dbl=dbl, repro=repro, mpe=mpe, mpo=mpo)
    # run to run: min p of each rerun, peaks paired to the main even fit
    mp_rr = np.array([paired(Pe, P)[3].min(1) for P in rr])               # (runs, N)
    allp = np.vstack([mpe[None], mp_rr])
    s['rr_sd'] = allp.std(0, ddof=1); s['rr_range'] = np.ptp(allp, 0)
    s['rr_curve'] = np.array([np.sqrt(((predict(P, fine) - predict(Pe, fine)) ** 2).mean(1)) / noise for P in rr]).max(0)
    if a in boots:
        s['boot_sd'] = np.array([paired(Pe, P)[3].min(1) for P in boots[a]]).std(0, ddof=1)
    if curve_A is not None:
        s['curve_vs_A'] = np.sqrt(((predict(Pe, fine) - curve_A) ** 2).mean(1)) / noise
        _, pa, _, px = paired(R['A']['even'], Pe); s['p_vs_A'] = np.abs(pa.min(1) - px.min(1))
    S[a] = s
    rho = spearmanr(mpe[dbl], mpo[dbl])[0]
    med_rep = np.median(((pe + po) / 2).min(1)[repro]) if repro.any() else np.nan
    ci = boot_ci(((pe + po) / 2).min(1)[repro]) if repro.sum() > 5 else (np.nan, np.nan)
    rows.append(dict(
        arm=a, label=LABEL[a], secs_even=float(R[a]['secs']),
        test_r2=float(np.median(s['test_r2'])),
        d_r2_vs_A=None if r2_A is None else float(np.median(s['test_r2'] - r2_A)),
        d_r2_ci=None if r2_A is None else boot_ci(s['test_r2'] - r2_A).tolist(),
        within_01=float(np.mean(s['excess'] <= 1e-3)), within_1=float(np.mean(s['excess'] <= 1e-2)),
        worse_5=float(np.mean(s['excess'] > 0.05)),
        n_double=int(dbl.sum()), spearman=float(rho), icc=float(icc(mpe[dbl], mpo[dbl])),
        med_abs_dp=float(np.median(np.abs(mpe - mpo)[dbl])),
        reproducible=int(repro.sum()), repro_share=float(repro.mean()),
        pop_median_repro=float(med_rep), pop_median_repro_ci=list(map(float, ci)),
        pop_median_all_double=float(np.median(mpe[dbl])),
        rr_median_sd=float(np.median(s['rr_sd'])), rr_share_range_gt_05=float(np.mean(s['rr_range'] > 0.5)),
        rr_curve_max_noise_units=float(np.median(s['rr_curve'])),
        boot_median_sd=float(np.median(s['boot_sd'])) if 'boot_sd' in s else None,
        curve_vs_A_median=float(np.median(s['curve_vs_A'])) if 'curve_vs_A' in s else None,
        curve_vs_A_share_lt1=float(np.mean(s['curve_vs_A'] < 1)) if 'curve_vs_A' in s else None,
        p_vs_A_share_gt05=float(np.mean(s['p_vs_A'] > 0.5)) if 'p_vs_A' in s else None,
        **({'lam': float(R[a]['lam'])} if a == 'F' else {}),
    ))
    if SYNTH:      # recovery of the known exponents (peaks paired to the truth's)
        _, pt, _, pf = paired(TRUTH, Pe)
        err = pf.min(1) - pt.min(1)
        rows[-1].update(rmse_minp=float(np.sqrt(np.mean(err ** 2))), bias_minp=float(np.median(err)),
                        within05=float(np.mean(np.abs(err) <= 0.5)),
                        rmse_minp_double=float(np.sqrt(np.mean(err[dbl] ** 2))),
                        repro_err=float(np.sqrt(np.mean(err[repro] ** 2))) if repro.any() else None,
                        truth_median_double=float(np.median(pt.min(1)[dbl])),
                        truth_median_repro=float(np.median(pt.min(1)[repro])) if repro.any() else None)

print(f'reference: best even-half MSE over {len(cands)} fits per cell; {N} neurons')
hdr = ('arm', 'secs', 'R2', 'dR2vsA', 'in0.1%', '>5%worse', 'double', 'rho', 'ICC', '|dp|', 'repro', 'popMed', 'rrSD', 'rr>0.5', 'bootSD', 'curveVsA', 'p>0.5vsA')
print(''.join(f'{h:>10s}' for h in hdr))
for r in rows:
    v = [r['arm'], r['secs_even'], r['test_r2'], r['d_r2_vs_A'], r['within_01'], r['worse_5'], r['n_double'], r['spearman'],
         r['icc'], r['med_abs_dp'], r['reproducible'], r['pop_median_repro'], r['rr_median_sd'], r['rr_share_range_gt_05'],
         r['boot_median_sd'], r['curve_vs_A_median'], r['p_vs_A_share_gt05']]
    print(''.join(f'{str(x):>10}' if isinstance(x, str) or x is None else (f'{x:>10d}' if isinstance(x, int) else f'{x:>10.4g}') for x in v))
if SYNTH:
    print(f'truth: population median min p = {np.median(TRUTH[:, 8:10].min(1)):.3f}')
    for r in rows:
        print(r['arm'], {k: round(r[k], 3) if isinstance(r[k], float) else r[k] for k in
                         ('rmse_minp', 'bias_minp', 'within05', 'rmse_minp_double', 'repro_err', 'truth_median_double',
                          'pop_median_all_double', 'truth_median_repro', 'pop_median_repro')})
for r in rows:
    print(r['arm'], 'dR2 CI', r['d_r2_ci'], 'pop median CI', r['pop_median_repro_ci'], 'curve<1 noise vs A', r['curve_vs_A_share_lt1'], r.get('lam', ''))
json.dump(rows, open(f'{OUT}/summary.json', 'w'), indent=1)

# ---------------------------------------------------------------- figure
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#666666', 'axes.labelcolor': '#191919', 'xtick.color': '#666666', 'ytick.color': '#666666'})
fig = plt.figure(figsize=(15, 10), dpi=150)
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)
y = np.arange(len(arms))[::-1]

def hbar(ax, vals, title, fmt='{:.0%}', ci=None):
    for yi, a, v in zip(y, arms, vals):
        ax.barh(yi, v, color=COLOR[a], height=0.6)
        ax.text(v, yi, ' ' + fmt.format(v), va='center', fontsize=9, color='#191919')
    if ci is not None:
        for yi, c in zip(y, ci): ax.plot(c, [yi, yi], color='#191919', lw=1)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[a] for a in arms], fontsize=9)
    ax.set_title(title, fontsize=11, loc='left'); ax.grid(axis='x', color='#e5e5e5', lw=0.8); ax.set_axisbelow(True)

hbar(fig.add_subplot(gs[0, 0]), [r['within_01'] for r in rows], 'Reaches the best known optimum\n(even-half MSE within 0.1%)')
hbar(fig.add_subplot(gs[0, 1]), [r['icc'] for r in rows], 'Split-half agreement of min(p)\n(ICC, even vs odd fits, double-peaked cells)', '{:.2f}')
hbar(fig.add_subplot(gs[0, 2]), [r['rr_share_range_gt_05'] for r in rows], 'Run-to-run inconsistency\n(cells whose min(p) moves > 0.5 on the same data)')
ax = fig.add_subplot(gs[1, 0])
if 'A' in R:
    hbar(ax, [r['d_r2_vs_A'] for r in rows], 'Held-out R², paired median minus Adam\n(fit even, test odd; 95% CI)', '{:+.4f}', [r['d_r2_ci'] for r in rows])
ax = fig.add_subplot(gs[1, 1])
hbar(ax, [r['secs_even'] for r in rows], 'CPU wall-clock, one 300-cell fit\n(4 threads; seconds, log scale)', '{:.0f} s'); ax.set_xscale('log')
ax = fig.add_subplot(gs[1, 2])
for a in arms:
    s = S[a]; ax.scatter(s['mpe'][s['dbl']], s['mpo'][s['dbl']], s=8, alpha=0.5, color=COLOR[a], label=a, edgecolors='none') if a in ('A', 'E') else None
ax.plot([0, 4], [0, 4], color='#666666', lw=1, ls='--'); ax.set_xlim(0, 4.1); ax.set_ylim(0, 4.1)
ax.set_xlabel('min(p), even-half fit'); ax.set_ylabel('min(p), odd-half fit'); ax.legend(title='arm', frameon=False, fontsize=9)
ax.set_title('Split-half min(p): Adam (A) vs bounds + starts (E)', fontsize=11, loc='left')
fig.suptitle(('SYNTHETIC (known p): ' if SYNTH else '') + 'Same 300 Stringer-2019 neurons, same halves: accuracy, optimum quality and self-consistency', fontsize=13, x=0.01, ha='left')
fig.savefig(f'{PNG}.png', bbox_inches='tight')

# example cells: the curves of A, B, C and E against the data, where A and E differ most and at the median
if 'A' in R and 'E' in R:
    d = S['E']['curve_vs_A']; pick = [int(np.argsort(d)[len(d) // 2]), int(np.argsort(d)[int(0.9 * len(d))]), int(np.argmax(d))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
    for ax, i in zip(axes, pick):
        ax.scatter(TH[i], HALF['even'][i], s=6, color='#191919', alpha=0.5, label='train (even)')
        ax.scatter(TH[i], HALF['odd'][i], s=6, color='#9c9c9c', alpha=0.5, label='test (odd)')
        for a in [x for x in 'ABCE' if x in R]:
            P = R[a]['even'][i:i + 1]; _, _, p = peaks(P)
            ax.plot(fine[0], predict(np.repeat(P, N, 0), fine)[0], color=COLOR[a], lw=2,
                    label=f'{a}: R² {S[a]["test_r2"][i]:.3f}, p {p[0, 0]:.2f}/{p[0, 1]:.2f}')
        ax.set_title(f'row {i}: curve gap A vs E = {d[i]:.2f} noise SE', fontsize=10, loc='left')
        ax.set_xticks([0, np.pi, 2 * np.pi]); ax.set_xticklabels(['0', '180', '360']); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f'{PNG}_examples.png', bbox_inches='tight')
print('saved analysis/consistency.png, consistency_examples.png and', f'{OUT}/summary.json')
