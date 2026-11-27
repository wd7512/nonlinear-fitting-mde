"""Same data, same question: the Adam notebook, our penalised notebook and four symbfitrs variants
fitted to the same 300 Stringer-2019 neurons, scored for accuracy, optimum quality and the
self-consistency of the fitted exponents.

Arms (each fitted on the even half, the odd half, and repeatedly on the even half with the
arbitrary choice the method makes varied, to measure run-to-run consistency):
  A  Adam notebook protocol (torch, CPU): validation-split early stopping, then all-bin refit.
     Run-to-run: the validation-split seed.
  B  our notebook as it stands: penalised toward the estimator, lambda = 1 (its own code, exec'd).
     Run-to-run: a perturbed start (which is also its penalty centre).
  C  plain symbfitrs LM to convergence on the clipped model.  Run-to-run: a perturbed start.
  D  symbfitrs with active-set bounds that replace the clips.  Run-to-run: a perturbed start.
  E  D, best of 8 starts (estimator, a p grid, 3 random jitters).  Run-to-run: the jitter seed.
  F  E plus a penalty toward the estimator, lambda chosen by 5-fold CV over the angle bins of
     the training half only.  Run-to-run: the jitter seed.
  boot_D / boot_E / boot_F: trial bootstrap of the even half (N_BOOT resamples).

usage (from the repo root):  PYTHONPATH=analysis python analysis/consistency.py ARM [ARM ...]
Each arm is cached in analysis/consistency/<arm>.npz; summarize_consistency.py reads them.
"""
import os, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import symbfitrs
from _neuro import ns, gen_gauss, estimator, EXPR, NAMES, MODEL, ORDER, INV, ROWS, trials, PENALISED_FIT_SRC

torch.set_num_threads(4)
OUT = 'analysis/consistency'
os.makedirs(OUT, exist_ok=True)
NUM_STEPS, LR, PATIENCE = ns['NUM_STEPS'], ns['LR'], ns['PATIENCE']
P_MIN, P_MAX, MIN_W, MAX_W = ns['P_MIN'], ns['P_MAX'], ns['MIN_WIDTH'], ns['MAX_WIDTH']
N_RR, N_STARTS, N_BOOT, BUDGET = 5, 8, int(os.environ.get('N_BOOT', 50)), 300.0
LAMBDAS = [0.0, 0.03, 0.1, 0.3, 1.0]
N = len(ROWS)
CELLS = [trials(r) for r in ROWS]                               # angles, [even (bins, 4), odd (bins, 4)]
TH = np.stack([c[0] for c in CELLS])                            # (N, bins)
SYNTH = os.environ.get('SYNTH') == '1'
if SYNTH:
    # Known truth: each neuron's arm-E fit on its even half (clipped into the model's ranges), with
    # both exponents redrawn uniformly in [0.7, 3]; trials = truth + Gaussian noise with that
    # neuron's per-bin trial SD (pooled over its 8 real trials). Same 4 trials per bin per half.
    OUT = 'analysis/consistency_synth'
    os.makedirs(OUT, exist_ok=True)
    _rng = np.random.default_rng(2026)
    TRUTH = np.load('analysis/consistency/E.npz')['even'].copy()
    TRUTH[:, 0] %= 2 * np.pi; TRUTH[:, 1:4] = TRUTH[:, 1:4].clip(min=0)
    TRUTH[:, 4:8] = TRUTH[:, 4:8].clip(MIN_W, MAX_W); TRUTH[:, 8:10] = _rng.uniform(0.7, 3.0, (N, 2))
    with torch.no_grad():
        _f = gen_gauss(torch.tensor(TH), torch.tensor(TRUTH)).squeeze(0).numpy()
    _sd = [np.concatenate(c[1], 1).std(1, ddof=1) for c in CELLS]
    CELLS = [(TH[i], [_f[i][:, None] + _sd[i][:, None] * _rng.standard_normal((len(_sd[i]), 4)) for _ in range(2)])
             for i in range(N)]
    np.save(f'{OUT}/truth.npy', TRUTH)
HALF = {h: np.stack([c[1][k].mean(1) for c in CELLS]) for k, h in enumerate(('even', 'odd'))}
NB = TH.shape[1]
P0 = {h: np.array([estimator(TH[i], HALF[h][i]) for i in range(N)], dtype=np.float64) for h in HALF}

# ---------------------------------------------------------------- shared pieces
def jitter(P, rng, s=1.0):
    """Perturb starts (k, 10) in the notebook's order; s=1 is a small nudge, s=3 a wide restart."""
    P = np.array(P, dtype=np.float64, copy=True)
    k = len(P)
    P[:, 0] = (P[:, 0] + rng.normal(0, 0.05 * s, k)) % (2 * np.pi)
    P[:, 1:4] *= np.exp(rng.normal(0, 0.1 * s, (k, 3)))
    P[:, 4:8] = np.clip(np.clip(P[:, 4:8], MIN_W, MAX_W) * np.exp(rng.normal(0, 0.2 * s, (k, 4))), MIN_W, MAX_W)
    P[:, 8:10] = np.clip(np.clip(P[:, 8:10], P_MIN, P_MAX) + rng.normal(0, 0.3 * s, (k, 2)), P_MIN, P_MAX)
    return P


def starts(p0, rng):
    """E/F: the estimator, the estimator with both exponents set to each of a grid, 3 wide jitters."""
    S = [p0]
    for p in (0.7, 1.4, 2.0, 3.0):
        q = p0.copy(); q[8:10] = p; S.append(q)
    S += list(jitter(np.repeat(p0[None], N_STARTS - len(S), 0), rng, 3.0))
    return np.array(S)


def score(P, half='even', idx=None):
    """MSE of the notebook's clipped model, float64, per cell."""
    idx = np.arange(N) if idx is None else idx
    with torch.no_grad():
        f = gen_gauss(torch.tensor(TH[idx]), torch.tensor(np.asarray(P, np.float64))).squeeze(0).numpy()
    return ((f - HALF[half][idx]) ** 2).mean(1)


def pmap(fn, items):
    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(fn, items))


# ---- the unclipped expression with bounds in place of the clips (as in bounds_unclipped_rust.py)
_W = '6.283185307179586'
_e = EXPR.replace(f'min(max(theta, 0), {_W})', 'theta')
for n in ('l1', 'r1', 'l2', 'r2'): _e = _e.replace(f'min(max({n}, 0.1), {_W})', n)
for n in ('p1', 'p2'): _e = _e.replace(f'min(max({n}, 0.1), 4.0)', n)
_i1, _i2 = _e.index(' + max(a1, 0)*'), _e.index(' + max(a2, 0)*')
FREE_EXPR = f'b + a1*{_e[_i1 + len(" + max(a1, 0)*"):_i2]} + a2*{_e[_i2 + len(" + max(a2, 0)*"):]}'
assert 'min(' not in FREE_EXPR and 'max(a1' not in FREE_EXPR and 'max(b' not in FREE_EXPR
FREE = symbfitrs.compile(FREE_EXPR)
F_ORDER = [list(NAMES).index(p) for p in FREE.params]; F_INV = np.argsort(F_ORDER)
BOUNDS = {'b': (0, None), 'a1': (0, None), 'a2': (0, None),
          **{k: (MIN_W, MAX_W) for k in ('l1', 'r1', 'l2', 'r2')}, 'p1': (P_MIN, P_MAX), 'p2': (P_MIN, P_MAX)}

# ---- penalised unclipped model: one extra data row per parameter, switched on by its indicator
SCALES = ns['PENALTY_SCALES']
WIDTHS = ('l1', 'r1', 'l2', 'r2')
TERMS = [f'log({n})' if n in WIDTHS else n for n in NAMES]
PVARS = ['x', 'm'] + [f'z{k}' for k in range(len(NAMES))]
PEN = symbfitrs.compile(f'm*({FREE_EXPR})' + ''.join(f' + z{k}*{t}' for k, t in enumerate(TERMS)), variables=PVARS)
P_ORDER = [list(NAMES).index(p) for p in PEN.params]; P_INV = np.argsort(P_ORDER)
STRENGTH = np.array([1.0 / SCALES[n] for n in NAMES])


def fit_bounded(th, y, p0, lam=0.0, centre=None):
    """One bounded LM fit of the unclipped model; lam > 0 adds the notebook's penalty toward centre."""
    p0 = np.asarray(p0, float).copy(); p0[0] %= 2 * np.pi
    if lam == 0.0:
        r = FREE.fit(th, y, p0=p0[F_ORDER], budget_ms=BUDGET, bounds=BOUNDS)
        return (r.params[F_INV] if r.params.size else p0), r
    c = np.asarray(centre, float).copy()
    c[4:8] = np.log(np.clip(c[4:8], MIN_W, MAX_W)); c[8:10] = np.clip(c[8:10], P_MIN, P_MAX)
    K, M = len(NAMES), len(th)
    X = np.zeros((K + 2, M + K)); X[0, :M], X[1, :M] = th, 1.0
    X[2:, M:] = np.diag(lam * STRENGTH)
    yy = np.r_[y, lam * STRENGTH * c]
    r = PEN.fit(X, yy, p0=p0[P_ORDER], budget_ms=BUDGET, bounds=BOUNDS)
    return (r.params[P_INV] if r.params.size else p0), r


def best_of(th, y, S, lam=0.0, centre=None):
    """Fit every start; keep the lowest penalised objective. Returns params, per-start losses."""
    fits = [fit_bounded(th, y, s, lam, centre) for s in S]
    losses = np.array([r.loss if r.params.size else np.inf for _, r in fits])
    j = int(np.argmin(losses))
    return fits[j][0], losses, np.array([p for p, _ in fits])


def clipped_fit(i, y, p0):
    r = MODEL.fit(TH[i], y, p0=np.asarray(p0)[ORDER], budget_ms=BUDGET)
    return r.params[INV] if r.params.size else p0

# ---------------------------------------------------------------- arms
def arm_A():
    """The Adam notebook's protocol, batched over cells (each cell its own angle grid)."""
    theta = torch.tensor(TH, dtype=torch.float32)
    gg = lambda th, P: gen_gauss(th, P).squeeze(0)

    def adam(Y, bins, P0, record):
        P = torch.tensor(P0, dtype=torch.float32).requires_grad_()
        opt = torch.optim.Adam([P], lr=LR); th = theta[:, bins]; y = torch.tensor(Y[:, bins], dtype=torch.float32)
        for step in range(NUM_STEPS):
            opt.zero_grad()
            (10 * ((gg(th, P) - y) ** 2).mean(1).mean()).backward()
            P.grad.nan_to_num_(0, 0, 0); opt.step()
            with torch.no_grad(): record(step, P.detach())

    def protocol(Y, seed):
        idx = np.random.default_rng(seed).permutation(NB); nv = NB // 5
        val2, val1, train, every = idx[:nv], idx[nv:2 * nv], idx[2 * nv:], np.arange(NB)
        est = lambda b: np.array([estimator(TH[i][b], Y[i][b]) for i in range(N)], dtype=np.float32)
        L1 = np.empty((NUM_STEPS, N), np.float32); L2 = np.empty_like(L1)
        y1, y2 = torch.tensor(Y[:, val1], dtype=torch.float32), torch.tensor(Y[:, val2], dtype=torch.float32)
        def rec(s, P):
            L1[s] = ((gg(theta[:, val1], P) - y1) ** 2).mean(1).numpy()
            L2[s] = ((gg(theta[:, val2], P) - y2) ** 2).mean(1).numpy()
        adam(Y, train, est(train), rec)
        best = np.nan_to_num(L1, nan=np.inf).argmin(0)                 # exactly the notebook's rule
        cand = np.clip(best[:, None] + PATIENCE[None], 0, NUM_STEPS - 1)
        stop = cand[np.arange(N), np.nan_to_num(L2, nan=np.inf)[cand, np.arange(N)[:, None]].argmin(1)]
        P0a = est(every); final = P0a.copy()
        def keep(s, P):
            m = stop == s
            if m.any(): final[m] = P.numpy()[m]
        adam(Y, every, P0a, keep)
        return final.astype(np.float64), stop

    out = {}
    t0 = time.time(); out['even'], out['stop_even'] = protocol(HALF['even'], ns['SEED']); out['secs'] = time.time() - t0
    print(f'A even: {out["secs"]:.0f}s', flush=True)
    out['odd'], out['stop_odd'] = protocol(HALF['odd'], ns['SEED'])
    out['rr'] = np.stack([protocol(HALF['even'], s)[0] for s in range(1, N_RR + 1)])
    return out


def arm_B():
    """Our notebook's penalised fit, its own code, one cell at a time (each cell its own grid)."""
    g = {'np': np, 'Counter': Counter, 'torch': torch, 'DEVICE': 'cpu', 'P_MIN': P_MIN, 'P_MAX': P_MAX,
         'MIN_WIDTH': MIN_W, 'MAX_WIDTH': MAX_W, 'PENALTY': ns['PENALTY'], 'PENALTY_SCALES': SCALES,
         'PENALTY_BUDGET_MS': ns['PENALTY_BUDGET_MS'], '_will_expression': ns['_will_expression']}
    exec(PENALISED_FIT_SRC, g)
    # Pool.fit_many inside it compiles one job per cell; feed one cell at a time with its own grid,
    # and allow the start to be overridden (for run-to-run) through the estimator hook.
    def run(Y, starts_=None):
        P = np.empty((N, 10))
        import io, contextlib
        for i in range(N):
            est = (lambda th, y, i=i: starts_[i]) if starts_ is not None else estimator
            g['MODELS'] = {'Gen. Gaussian': (gen_gauss, est)}
            with contextlib.redirect_stdout(io.StringIO()):
                P[i] = g['penalised_fit']('Gen. Gaussian', TH[i], Y[i][None]).numpy()[0]
        return P
    out = {}
    t0 = time.time(); out['even'] = run(HALF['even']); out['secs'] = time.time() - t0
    out['odd'] = run(HALF['odd'])
    # the notebook uses one vector as both start and penalty centre, so this perturbs both
    out['rr'] = np.stack([run(HALF['even'], jitter(P0['even'], np.random.default_rng(100 + s))) for s in range(N_RR)])
    return out


def arm_C():
    out = {}
    run = lambda h, S: np.array(pmap(lambda i: clipped_fit(i, HALF[h][i], S[i]), range(N)))
    t0 = time.time(); out['even'] = run('even', P0['even']); out['secs'] = time.time() - t0
    out['odd'] = run('odd', P0['odd'])
    out['rr'] = np.stack([run('even', jitter(P0['even'], np.random.default_rng(100 + s))) for s in range(N_RR)])
    return out


def arm_D():
    out = {}
    run = lambda h, S: np.array(pmap(lambda i: fit_bounded(TH[i], HALF[h][i], S[i])[0], range(N)))
    t0 = time.time(); out['even'] = run('even', P0['even']); out['secs'] = time.time() - t0
    out['odd'] = run('odd', P0['odd'])
    out['rr'] = np.stack([run('even', jitter(P0['even'], np.random.default_rng(100 + s))) for s in range(N_RR)])
    return out


def multi(h, seed, lam=0.0, Y=None, idx=None):
    """E/F on one half (or bootstrap curves Y): best of N_STARTS per cell."""
    Y = HALF[h] if Y is None else Y
    centre = P0[h] if Y is HALF[h] else np.array([estimator(TH[i], Y[i]) for i in range(N)])
    def one(i):
        S = starts(centre[i], np.random.default_rng([seed, i]))
        return best_of(TH[i], Y[i], S, lam, centre[i])
    res = pmap(one, range(N))
    return np.array([r[0] for r in res]), np.array([r[1] for r in res]), np.array([r[2] for r in res])


def arm_E():
    out = {}
    t0 = time.time(); out['even'], out['start_loss'], out['start_params'] = multi('even', 0); out['secs'] = time.time() - t0
    out['odd'] = multi('odd', 0)[0]
    out['rr'] = np.stack([multi('even', s)[0] for s in range(1, N_RR + 1)])
    return out


def choose_lambda():
    """5-fold CV over angle bins of the EVEN half only: fit on 4/5 of the bins (estimator start and
    penalty centre from those bins), score on the held-out 1/5. One lambda for the population."""
    folds = np.array_split(np.random.default_rng(11).permutation(NB), 5)
    cv = np.zeros((len(LAMBDAS), N))
    for k, test in enumerate(folds):
        tr = np.setdiff1d(np.arange(NB), test)
        def one(i):
            th, y = TH[i][tr], HALF['even'][i][tr]
            c = estimator(th, y)
            S = starts(c, np.random.default_rng([99, i, k]))
            row = []
            for lam in LAMBDAS:
                P = best_of(th, y, S, lam, c)[0]
                with torch.no_grad():
                    f = gen_gauss(torch.tensor(TH[i][test]), torch.tensor(P[None])).squeeze(0).numpy()[0]
                row.append(((f - HALF['even'][i][test]) ** 2).mean())
            return row
        cv += np.array(pmap(one, range(N))).T
        print(f'  CV fold {k + 1}/5', flush=True)
    rel = cv / cv.min(0, keepdims=True)                         # per-cell loss relative to its best lambda
    return LAMBDAS[int(np.argmin(rel.mean(1)))], cv


def arm_F():
    out = {}
    t0 = time.time(); lam, out['cv'] = choose_lambda(); out['cv_secs'] = time.time() - t0
    out['lam'] = lam; print(f'F: CV picked lambda = {lam}', flush=True)
    t0 = time.time(); out['even'] = multi('even', 0, lam)[0]; out['secs'] = time.time() - t0
    out['odd'] = multi('odd', 0, lam)[0]
    out['rr'] = np.stack([multi('even', s, lam)[0] for s in range(1, N_RR + 1)])
    return out


def boot(kind):
    """Resample the 4 even-half trials in every bin with replacement; refit."""
    lam = float(np.load(f'{OUT}/F.npz')['lam']) if kind == 'F' else 0.0
    even = [c[1][0] for c in CELLS]
    rng = np.random.default_rng(5)
    out = np.empty((N_BOOT, N, 10)); t0 = time.time()
    for b in range(N_BOOT):
        Y = np.stack([t[np.arange(len(t))[:, None], rng.integers(0, t.shape[1], t.shape)].mean(1) for t in even])
        if kind == 'D':
            est = np.array([estimator(TH[i], Y[i]) for i in range(N)])
            out[b] = np.array(pmap(lambda i: fit_bounded(TH[i], Y[i], est[i])[0], range(N)))
        else:
            out[b] = multi('even', 1000 + b, lam, Y=Y)[0]
        if b % 10 == 9: print(f'  boot {kind} {b + 1}/{N_BOOT} {time.time() - t0:.0f}s', flush=True)
    return {'P': out, 'secs': time.time() - t0}


ARMS = {'A': arm_A, 'B': arm_B, 'C': arm_C, 'D': arm_D, 'E': arm_E, 'F': arm_F,
        'boot_D': lambda: boot('D'), 'boot_E': lambda: boot('E'), 'boot_F': lambda: boot('F')}

if __name__ == '__main__':
    for a in sys.argv[1:]:
        print(f'arm {a}', flush=True); t0 = time.time()
        res = ARMS[a]()
        np.savez(f'{OUT}/{a}.npz', rows=ROWS, **res)
        print(f'arm {a} saved in {time.time() - t0:.0f}s', flush=True)
