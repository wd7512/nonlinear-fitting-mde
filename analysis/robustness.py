"""Is the single-neuron picture robust? The jacob_tuning validation/refit protocol over many
Stringer-2019 neurons: Adam (batched over cells, as in the notebook) vs symbfitrs through
Pool.fit_many(trace=True). Reports where validation loss bottoms out, what the stopping rule
selects, held-out R^2 under several stopping rules, and sensitivity to the validation split.

usage: python analysis/robustness.py N_CELLS OUT.json [SEEDS]
"""
import glob, json, sys, time
import numpy as np
import torch
import symbfitrs

N_CELLS, OUT = int(sys.argv[1]), sys.argv[2]
SPLIT_SEEDS = [int(s) for s in sys.argv[3].split(',')] if len(sys.argv) > 3 else [7]
UP = glob.glob('/root/.claude/uploads/*/')[0]
torch.set_num_threads(4)

adam_nb = json.load(open(UP + '9d73f0a9-jacob_tuning_fits.ipynb'))
will_nb = json.load(open(UP + '06379ac6-jacob_tuning_will_fits.ipynb'))
src = lambda nb, i: ''.join(nb['cells'][i]['source'])
ns = {'np': np, 'torch': torch}
exec(src(adam_nb, 17), ns); exec(src(adam_nb, 18), ns)
w = src(will_nb, 22); exec(w[w.index('def _will_expression'):w.index('def will_fit')], ns)
exec(src(adam_nb, 20).split('def mse')[0], ns)
gen_gauss, estimator = ns['gen_gauss'], ns['parameter_estimator_bz15']
NUM_STEPS, LR, PATIENCE = ns['NUM_STEPS'], ns['LR'], ns['PATIENCE']
BUDGET_MS = 1000.0

d = np.load('data/external/stringer2019/gratings.npz')
X, Y, ids = d['x'], d['y'], d['cell_ids']
BIN = 8


def binned(r):
    o = np.argsort(X[r]); th, y = X[r][o], Y[r][o]; n = len(th) // BIN
    th, y = th[:n * BIN].reshape(n, BIN), y[:n * BIN].reshape(n, BIN)
    R = np.stack([y[:, 0::2].mean(1), y[:, 1::2].mean(1)])
    return th.mean(1), R / np.sqrt((R ** 2).mean(1, keepdims=True))


def screens(r):
    th, R = binned(r); n = R.shape[1]
    mu, var = R.mean(0), R.var(0, ddof=1)
    S2 = ((mu - mu.mean()) ** 2).mean() - (n - 1) / n**2 * (var / 2).sum(); V2 = (var / 2).sum() / n
    yc = np.clip(Y[r], 0, None)
    return S2 / (S2 + V2 + 1e-12), abs((yc * np.exp(2j * X[r])).sum()) / yc.sum()


sc = np.array([screens(r) for r in range(len(X))])
passing = np.where((sc[:, 0] > 0.6) & (sc[:, 1] > 0.3))[0]
rows = np.sort(np.random.default_rng(0).choice(passing, min(N_CELLS, len(passing)), replace=False))
print(f'{len(passing)} of {len(X)} cells pass SF>0.6, C>0.3; using {len(rows)} random ones', flush=True)
TH, Rr = map(np.stack, zip(*[binned(r) for r in rows]))   # each cell has its own angle grid: (cells, bins), (cells, 2, bins)
Ytr, Yte = Rr[:, 0], Rr[:, 1]
theta = torch.tensor(TH, dtype=torch.float32)   # (cells, bins)
n_bins = TH.shape[1]
r2 = lambda y, p: 1 - ((y - p) ** 2).sum(-1) / ((y - y.mean(-1, keepdims=True)) ** 2).sum(-1)


def gg(th, P):   # th (cells, bins), P (cells, 10): the notebook's model with a per-cell angle grid
    return gen_gauss(th, P).squeeze(0)


def model_np(P, bins=None, cell=None):   # batched over cells, or P (steps, 10) for one cell
    sl = slice(None) if bins is None else bins
    with torch.no_grad():
        P = torch.as_tensor(P, dtype=torch.float32)
        if cell is not None:
            return gen_gauss(theta[cell][sl], P).numpy()
        return gg(theta[:, sl], P).numpy()


_P = np.array([estimator(TH[i], Rr[i, 0]) for i in range(len(rows))], dtype=np.float32)
_chk = model_np(_P)
for i in range(3):   # the batched per-cell call equals the one-cell call
    assert np.allclose(_chk[i], model_np(_P[i:i + 1], cell=i)[0], atol=1e-6)


def estimates(bins):
    return np.array([estimator(TH[i][bins], Ytr[i][bins]) for i in range(len(rows))], dtype=np.float32)


def adam(bins, P0, record=None):
    P = torch.tensor(P0).requires_grad_()
    opt = torch.optim.Adam([P], lr=LR); th = theta[:, bins]; y = torch.tensor(Ytr[:, bins], dtype=torch.float32)
    for step in range(NUM_STEPS):
        opt.zero_grad()
        (10 * ((gg(th, P) - y) ** 2).mean(1).mean()).backward()
        P.grad.nan_to_num_(0, 0, 0); opt.step()
        if record: record(step + 1, P.detach())
    return P.detach().numpy()


def pick(l1, l2):     # (steps, cells) -> val1-best, notebook stop (best + patience, then val2)
    best = l1.argmin(0)
    cand = np.clip(best[:, None] + PATIENCE[None], 0, len(l1) - 1)
    c = np.arange(l1.shape[1])[:, None]
    return best, cand[np.arange(l1.shape[1]), l2[cand, c].argmin(1)]


expr, names = ns['_will_expression']('Gen. Gaussian')
model = symbfitrs.compile(expr); order = [names.index(p) for p in model.params]; inv = np.argsort(order)
out = dict(rows=rows.tolist(), cell_ids=ids[rows].tolist(), sf=sc[rows, 0].tolist(), c=sc[rows, 1].tolist(), splits={})

for seed in SPLIT_SEEDS:
    idx = np.random.default_rng(seed).permutation(n_bins); nv = n_bins // 5
    val2, val1, train, every = idx[:nv], idx[nv:2 * nv], idx[2 * nv:], np.arange(n_bins)
    P0_tr, P0_all = estimates(train), estimates(every)
    res = {'start_r2': r2(Yte, model_np(P0_all)).tolist()}
    # ---- Adam: train phase with per-step val losses, then all-angle refit keeping the stop step
    t0 = time.time()
    L1 = np.empty((NUM_STEPS + 1, len(rows)), np.float32); L2 = np.empty_like(L1)
    yv1, yv2 = Ytr[:, val1], Ytr[:, val2]
    def rec(s, P):
        L1[s] = ((gg(theta[:, val1], P).numpy() - yv1) ** 2).mean(1)
        L2[s] = ((gg(theta[:, val2], P).numpy() - yv2) ** 2).mean(1)
    L1[0] = ((model_np(P0_tr, val1) - yv1) ** 2).mean(1); L2[0] = ((model_np(P0_tr, val2) - yv2) ** 2).mean(1)
    adam(train, P0_tr, rec)
    best, stop = pick(L1, L2)
    stops = {'notebook': stop, 'val1_best': best, 'no_stop': np.full(len(rows), NUM_STEPS)}
    keep = {k: np.array(P0_all) for k in stops}
    def keeper(s, P):
        for k, st in stops.items():
            m = st == s
            if m.any(): keep[k][m] = P.numpy()[m]
    adam(every, P0_all, keeper)
    for k in stops: keep[k][stops[k] == 0] = P0_all[stops[k] == 0]
    res['adam'] = dict(secs=time.time() - t0, best=best.tolist(), stop=stop.tolist(),
                       r2={k: r2(Yte, model_np(keep[k])).tolist() for k in keep},
                       val1_gain=((L1[0] - L1.min(0)) / L1[0]).tolist())
    print(f'seed {seed}: Adam done in {res["adam"]["secs"]:.0f}s', flush=True)
    # ---- symbfitrs via Pool.fit_many(trace=True)
    pool = symbfitrs.Pool()
    def run(bins, P0):
        jobs = [(model, TH[i][bins], Ytr[i, bins], P0[i][order]) for i in range(len(rows))]
        return pool.fit_many(jobs, budget_ms=BUDGET_MS, progress=False, trace=True, max_steps=NUM_STEPS)
    t0 = time.time(); R1 = run(train, P0_tr)
    S1 = [np.array([((model_np(R.trace[:, inv], b, cell=i) - Ytr[i, b]) ** 2).mean(1) for b in (val1, val2)]) for i, R in enumerate(R1)]
    sb, ss = [], []
    for l in S1:
        l1, l2 = l[0][:, None], l[1][:, None]
        b, s = pick(l1, l2); sb.append(int(b[0])); ss.append(int(s[0]))
    R2 = run(every, P0_all)
    sstops = {'notebook': np.array(ss), 'val1_best': np.array(sb), 'no_stop': np.array([r.steps for r in R2])}
    sym_r2 = {}
    for k, st in sstops.items():
        P = np.array([R2[i].trace[min(st[i], len(R2[i].trace) - 1)][inv] for i in range(len(rows))])
        sym_r2[k] = r2(Yte, model_np(P)).tolist()
    res['sym'] = dict(secs=time.time() - t0, best=sb, stop=ss, steps=[r.steps for r in R1], refit_steps=[r.steps for r in R2],
                      status=[r.status for r in R1], refit_status=[r.status for r in R2], ms=[r.time_ms for r in R1],
                      refit_ms=[r.time_ms for r in R2], r2=sym_r2,
                      val1_gain=[float((l[0][0] - l[0].min()) / l[0][0]) for l in S1])
    # trace prefix property, checked on real fits through the capped API
    chk = 0
    for i in range(0, len(rows), max(1, len(rows) // 12)):
        k = min(stop_i := int(ss[i]), len(R2[i].trace) - 1)
        c = model.fit(TH[i][every], Ytr[i, every], p0=P0_all[i][order], budget_ms=BUDGET_MS, trace=True, max_steps=k)
        if R2[i].status != 'deadline': assert np.array_equal(c.trace, R2[i].trace[:k + 1]), f'cell {i}'; chk += 1
    res['prefix_checked'] = chk
    print(f'seed {seed}: symbfitrs done in {res["sym"]["secs"]:.0f}s; prefix property verified on {chk} cells', flush=True)
    out['splits'][str(seed)] = res
json.dump(out, open(OUT, 'w'))
print('saved', OUT)
