"""Quick idea loop on a 60-neuron subset (minutes, not hours): generalisation and parameter
recovery of the estimator alone, Adam with and without early stopping, and symbfitrs variants
(bounds + 8 starts through Pool.fit_many) with different model limits and width parametrisations.

Datasets: the real even/odd halves, and synthetic halves from known parameters (truth = each
neuron's arm-E fit with both exponents redrawn in [0.7, 3]; noise = that neuron's per-bin trial SD).

Generalisation yardsticks, per method:
  test R2      fit even, score odd (fresh trials at the same angles)
  held-out angles  fit on 80% of the bins (Adam: its own protocol inside those 80%), score MSE on
               the other 20% of angles, relative to the estimator; < 1 is better
  df_eff       effective parameters from the optimism: n (MSE_test - MSE_train) / (2 sigma^2);
               about 10 means the 10-parameter model fits as much noise as a plain LSQ fit should
  curve err    synthetic only: RMS distance from the TRUE curve in units of the noise SE of the
               mean curve; the direct "is it fitting noise" measure
Parameters: split-half ICC of min p, run-to-run share moving > 0.5, share with p at its limit,
synthetic RMSE / median error of min p.

usage: PYTHONPATH=analysis python analysis/explore.py [name-substring ...]   (default: all)
Adam is run once and cached in analysis/explore/adam.npz (delete it to rerun).
"""
import os, sys, time
import numpy as np
import torch
import symbfitrs
import consistency as C

N_SUB, STARTS, MS_PER_START = 60, 8, 100.0
OUT = 'analysis/explore'; os.makedirs(OUT, exist_ok=True)
SUB = np.arange(0, C.N, C.N // N_SUB)[:N_SUB]
TH = C.TH[SUB]; NB = TH.shape[1]
LN2 = 0.6931471805599453
HOLD = np.sort(np.random.default_rng(99).permutation(NB)[:NB // 5])     # held-out angles (20%)
KEEP = np.setdiff1d(np.arange(NB), HOLD)

# ---------------------------------------------------------------- data
_rng = np.random.default_rng(2026)
TRUTH = np.load('analysis/consistency/E.npz')['even'][SUB].copy()
TRUTH[:, 0] %= 2 * np.pi; TRUTH[:, 1:4] = TRUTH[:, 1:4].clip(min=0)
TRUTH[:, 4:8] = TRUTH[:, 4:8].clip(0.1, 2 * np.pi); TRUTH[:, 8:10] = _rng.uniform(0.7, 3.0, (N_SUB, 2))


def gg(P, th=TH):
    with torch.no_grad():
        return C.gen_gauss(torch.tensor(th, dtype=torch.float64), torch.tensor(np.asarray(P, np.float64))).squeeze(0).numpy()


F_TRUE = gg(TRUTH)
_sd_real = {h: np.stack([C.CELLS[i][1][k].std(1, ddof=1) / 2 for i in SUB]) for k, h in enumerate(('even', 'odd'))}
_sd_syn = np.stack([np.concatenate(C.CELLS[i][1], 1).std(1, ddof=1) / 2 for i in SUB])   # SE of a 4-trial mean
DATA = {'real': {h: C.HALF[h][SUB] for h in ('even', 'odd')},
        'synth': {h: F_TRUE + _sd_syn * _rng.standard_normal(F_TRUE.shape) for h in ('even', 'odd')}}
SIGMA2 = {'real': (_sd_real['even'] ** 2).mean(1), 'synth': (_sd_syn ** 2).mean(1)}       # noise var of a mean, per cell
EST = lambda Y, bins=slice(None): np.array([C.estimator(TH[i][bins], Y[i][bins]) for i in range(N_SUB)])

# ---------------------------------------------------------------- symbfitrs variants
def peak(off, l, r, p, hwhm):
    d = f'(2*atan(tan((x - (theta + {off}))/2)))'
    w = f'({r} + ({l} - {r})*max(-sign({d}), 0))'
    return f'exp(-{LN2 if hwhm else 0.5}*(max(abs({d}), 1e-12)/{w})^{p})'


def variant(p_max=4.0, w_max=2 * np.pi, hwhm=False, starts=STARTS, w_min=None):
    m = symbfitrs.compile(f'b + a1*{peak(0, "l1", "r1", "p1", hwhm)} + a2*{peak("pi", "l2", "r2", "p2", hwhm)}')
    w_min = (0.05 if hwhm else 0.1) if w_min is None else w_min
    bounds = {'b': (0, None), 'a1': (0, None), 'a2': (0, None),
              **{k: (w_min, w_max) for k in ('l1', 'r1', 'l2', 'r2')}, 'p1': (0.1, p_max), 'p2': (0.1, p_max)}
    return dict(kind='sym', model=m, bounds=bounds, p_max=p_max, w_max=w_max, w_min=w_min, hwhm=hwhm, starts=starts,
                order=[list(C.NAMES).index(p) for p in m.params])


def clip_v(P, v):
    P = np.array(P, float); P[:, 0] %= 2 * np.pi
    P[:, 4:8] = P[:, 4:8].clip(v['w_min'], v['w_max']); P[:, 8:10] = P[:, 8:10].clip(0.1, v['p_max'])
    return P


def to_variant(P, v):
    """Estimator output (sigma widths) -> the variant's widths (hwhm: h = sigma (2 ln2)^(1/p))."""
    P = np.array(P, float); P[:, 8:10] = P[:, 8:10].clip(0.1, v['p_max'])
    if v['hwhm']:
        for k, j in ((4, 8), (5, 8), (6, 9), (7, 9)):
            P[:, k] *= (2 * LN2) ** (1 / P[:, j])
    return clip_v(P, v)


def starts(p0, rng, v):
    S = [p0]
    for p in (0.7, 1.4, 2.0, 3.0):
        q = p0.copy(); q[8:10] = p
        if not v['hwhm']:     # sigma widths: move them so the half-width stays put as p changes
            q[4:6] *= (2 * LN2) ** (1 / p0[8] - 1 / p); q[6:8] *= (2 * LN2) ** (1 / p0[9] - 1 / p)
        S.append(q)
    while len(S) < max(v['starts'], 5):
        S.append(C.jitter(p0[None], rng, 3.0)[0])
    return clip_v(np.array(S[:v['starts']]), v)


def fit_sym(v, Y, bins, seed):
    rng = np.random.default_rng(seed)
    P0 = to_variant(EST(Y, bins), v)
    jobs = [(v['model'], TH[i][bins], Y[i][bins], starts(P0[i], rng, v)[:, v['order']]) for i in range(N_SUB)]
    res = symbfitrs.Pool(threads=4).fit_many(jobs, budget_ms=MS_PER_START * v['starts'], bounds=v['bounds'], progress=False)
    inv = np.argsort(v['order'])
    return np.array([r.params[inv] for r in res]), dict(conv=np.mean([r.status == 'converged' for r in res]),
                                                        ms=np.median([r.time_ms for r in res]))


def predict(m, P):
    if m['kind'] == 'sym':
        return np.array([m['model'].predict(TH[i], P[i][m['order']]) for i in range(N_SUB)])
    return gg(P)


def p_at_max(m, P):
    return np.mean((P[:, 8:10] >= m.get('p_max', 4.0) - 1e-6).any(1))


# ---------------------------------------------------------------- Adam (the notebook protocol), cached
def adam_protocol(Y, bins, seed):
    """The notebook's procedure restricted to `bins`: val1/val2 split inside them, train, pick the
    stop, refit on all of `bins`. Returns (early-stopped, no-stop) parameters."""
    theta = torch.tensor(TH[:, bins], dtype=torch.float32); y = torch.tensor(Y[:, bins], dtype=torch.float32)
    nb = len(bins); idx = np.random.default_rng(seed).permutation(nb); nv = nb // 5
    val2, val1, train = idx[:nv], idx[nv:2 * nv], idx[2 * nv:]
    f = lambda th, P: C.gen_gauss(th, P).squeeze(0)
    def run(sub, P0, rec):
        P = torch.tensor(P0, dtype=torch.float32).requires_grad_(); opt = torch.optim.Adam([P], lr=C.LR)
        for s in range(C.NUM_STEPS):
            opt.zero_grad(); (10 * ((f(theta[:, sub], P) - y[:, sub]) ** 2).mean(1).mean()).backward()
            P.grad.nan_to_num_(0, 0, 0); opt.step()
            with torch.no_grad(): rec(s, P.detach())
    est = lambda sub: np.array([C.estimator(TH[i][bins][sub], Y[i][bins][sub]) for i in range(N_SUB)], np.float32)
    L1 = np.empty((C.NUM_STEPS, N_SUB), np.float32); L2 = np.empty_like(L1)
    def rec(s, P):
        L1[s] = ((f(theta[:, val1], P) - y[:, val1]) ** 2).mean(1).numpy(); L2[s] = ((f(theta[:, val2], P) - y[:, val2]) ** 2).mean(1).numpy()
    run(train, est(train), rec)
    best = np.nan_to_num(L1, nan=np.inf).argmin(0)
    cand = np.clip(best[:, None] + C.PATIENCE[None], 0, C.NUM_STEPS - 1)
    stop = cand[np.arange(N_SUB), np.nan_to_num(L2, nan=np.inf)[cand, np.arange(N_SUB)[:, None]].argmin(1)]
    allb = np.arange(nb); P0 = est(allb); early = P0.copy(); last = P0.copy()
    def keep(s, P):
        m = stop == s
        if m.any(): early[m] = P.numpy()[m]
        if s == C.NUM_STEPS - 1: last[:] = P.numpy()
    run(allb, P0, keep)
    return early.astype(float), last.astype(float)


def adam_cache():
    path = f'{OUT}/adam.npz'
    if os.path.exists(path):
        return dict(np.load(path))
    torch.set_num_threads(4); out = {}; t0 = time.time()
    for d in DATA:
        for tag, Y, bins, seed in (('even', DATA[d]['even'], np.arange(NB), 7), ('odd', DATA[d]['odd'], np.arange(NB), 7),
                                   ('rr', DATA[d]['even'], np.arange(NB), 1), ('hold', DATA[d]['even'], KEEP, 7)):
            out[f'{d}_{tag}_early'], out[f'{d}_{tag}_last'] = adam_protocol(Y, bins, seed)
            print(f'  adam {d} {tag}: {time.time() - t0:.0f}s', flush=True)
    np.savez(path, **out)
    return out


# ---------------------------------------------------------------- methods
METHODS = {
    'estimator only': dict(kind='est'),
    'Adam, early stop': dict(kind='adam', which='early'),
    'Adam, no stop': dict(kind='adam', which='last'),
    'E: bounds+8 starts': variant(),
    'E, 1 start': variant(starts=1),
    'E, p<=10': variant(p_max=10.0),
    'E, w<=pi': variant(w_max=np.pi),
    'E, hwhm widths': variant(hwhm=True, w_max=np.pi),
    'E, hwhm, p<=10': variant(hwhm=True, w_max=np.pi, p_max=10.0),
    'E, hwhm, w<=2pi': variant(hwhm=True),
    'E, hwhm, w>=0.1': variant(hwhm=True, w_min=0.1),
    'E, hwhm, w>=0.2': variant(hwhm=True, w_min=0.2),
    'E, sigma w>=0.2': variant(w_min=0.2),
}
_A = TRUTH[:, 2:4]
DOUBLE = _A.min(1) / _A.max(1) >= 0.3            # synthetic cells whose weaker peak is clearly real
STRONG = _A.argmax(1)                            # the truth's dominant peak


def fits(m, d):
    """Parameters for: even, odd, run-to-run rerun of even, and the held-out-angle fit."""
    D = DATA[d]
    if m['kind'] == 'est':
        P = {'even': EST(D['even']), 'odd': EST(D['odd']), 'hold': EST(D['even'], KEEP)}
        P['rr'] = P['even']; return P, {}
    if m['kind'] == 'adam':
        A = adam_cache(); return {t: A[f'{d}_{t}_{m["which"]}'] for t in ('even', 'odd', 'rr', 'hold')}, {}
    allb = np.arange(NB)
    Pe, info = fit_sym(m, D['even'], allb, 0)
    return dict(even=Pe, odd=fit_sym(m, D['odd'], allb, 0)[0], rr=fit_sym(m, D['even'], allb, 1)[0],
                hold=fit_sym(m, D['even'], KEEP, 0)[0]), info


def paired_minp(Pa, Pb):
    swap = np.abs(np.arctan2(np.sin(Pa[:, 0] - Pb[:, 0]), np.cos(Pa[:, 0] - Pb[:, 0]))) > np.pi / 2
    pb = np.where(swap[:, None], Pb[:, 8:10][:, ::-1], Pb[:, 8:10]).clip(0.1, None)
    return pb.min(1)


def icc(x, y):
    dd = np.stack([x, y], 1); gm = dd.mean()
    msb = 2 * ((dd.mean(1) - gm) ** 2).sum() / (len(dd) - 1); msw = ((dd - dd.mean(1, keepdims=True)) ** 2).sum() / len(dd)
    return (msb - msw) / (msb + msw)


def evaluate(name, m):
    t0 = time.time(); out = {}
    for d in DATA:
        P, info = fits(m, d); D = DATA[d]
        fe, fh = predict(m, P['even']), predict(m, P['hold'])
        mse_tr = ((fe - D['even']) ** 2).mean(1); mse_te = ((fe - D['odd']) ** 2).mean(1)
        r2 = 1 - ((D['odd'] - fe) ** 2).sum(1) / ((D['odd'] - D['odd'].mean(1, keepdims=True)) ** 2).sum(1)
        hold = ((fh[:, HOLD] - D['even'][:, HOLD]) ** 2).mean(1)
        mpe = np.clip(P['even'][:, 8:10], 0.1, None).min(1)
        o = dict(r2=np.median(r2), hold=hold, df=np.median(NB * (mse_te - mse_tr) / (2 * SIGMA2[d])),
                 icc=icc(mpe, paired_minp(P['even'], P['odd'])), rr=np.mean(np.abs(paired_minp(P['even'], P['rr']) - mpe) > 0.5),
                 at_p=p_at_max(m, P['even']), **info)
        if d == 'synth':
            o['curve'] = np.median(np.sqrt(((fe - F_TRUE) ** 2).mean(1) / SIGMA2[d]))
            err = mpe - paired_minp(P['even'], TRUTH)
            o.update(rmse=np.sqrt(np.mean(err ** 2)), bias=np.median(err), within=np.mean(np.abs(err) <= 0.5),
                     rmse_dbl=np.sqrt(np.mean(err[DOUBLE] ** 2)))
            # the dominant peak's own exponent, with the fit's peaks paired to the truth's
            swap = np.abs(np.arctan2(np.sin(TRUTH[:, 0] - P['even'][:, 0]), np.cos(TRUTH[:, 0] - P['even'][:, 0]))) > np.pi / 2
            pf = np.where(swap[:, None], P['even'][:, 8:10][:, ::-1], P['even'][:, 8:10]).clip(0.1, None)
            es = pf[np.arange(N_SUB), STRONG] - TRUTH[np.arange(N_SUB), 8 + STRONG]
            o.update(rmse_strong=np.sqrt(np.mean(es ** 2)), within_strong=np.mean(np.abs(es) <= 0.5))
        out[d] = o
    out['secs'] = time.time() - t0
    return out


if __name__ == '__main__':
    names = [n for n in METHODS if not sys.argv[1:] or any(a in n for a in sys.argv[1:])]
    if 'estimator only' not in names: names = ['estimator only'] + names       # the held-out-angle reference
    R = {n: evaluate(n, METHODS[n]) for n in names}
    ref = R['estimator only']
    print(f'\n{N_SUB} neurons; held-out angles = {len(HOLD)} of {NB} bins; symbfitrs: {MS_PER_START:.0f} ms per start')
    print('REAL DATA')
    print(f'{"method":22s}{"testR2":>7s}{"heldout":>8s}{"df_eff":>7s}{"ICC":>6s}{"rr>.5":>6s}{"p@max":>6s}{"conv":>6s}{"secs":>6s}')
    for n in names:
        o = R[n]['real']
        print(f'{n:22s}{o["r2"]:7.3f}{np.median(o["hold"] / ref["real"]["hold"]):8.3f}{o["df"]:7.1f}{o["icc"]:6.2f}'
              f'{o["rr"]:6.0%}{o["at_p"]:6.0%}{o.get("conv", np.nan):6.0%}{R[n]["secs"]:6.0f}')
    print('SYNTHETIC (known truth; curve err in noise-SE units, lower = less noise fitted)')
    print(f'{"method":22s}{"testR2":>7s}{"heldout":>8s}{"df_eff":>7s}{"curve":>7s}{"RMSEp":>7s}{"bias":>6s}{"in.5":>6s}{"dblRMSE":>8s}{"strRMSE":>8s}{"str.5":>6s}{"rr>.5":>6s}{"p@max":>6s}')
    for n in names:
        o = R[n]['synth']
        print(f'{n:22s}{o["r2"]:7.3f}{np.median(o["hold"] / ref["synth"]["hold"]):8.3f}{o["df"]:7.1f}{o["curve"]:7.3f}'
              f'{o["rmse"]:7.2f}{o["bias"]:+6.2f}{o["within"]:6.0%}{o["rmse_dbl"]:8.2f}{o["rmse_strong"]:8.2f}{o["within_strong"]:6.0%}{o["rr"]:6.0%}{o["at_p"]:6.0%}')
    print(f'({DOUBLE.sum()} of {N_SUB} synthetic cells are clearly double-peaked)')
    np.save(f'{OUT}/last_run.npy', {n: R[n] for n in names}, allow_pickle=True)
