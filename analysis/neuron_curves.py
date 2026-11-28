"""MSE learning curves, Adam vs symbfitrs, for one Stringer-2019 neuron, following the
jacob_tuning notebooks' validation-and-refit protocol (generalised-Gaussian / BZ15 model)."""
import glob, json, sys, time
import numpy as np
import torch

REPO = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))  # the repo root
CELL = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else None
OUT = sys.argv[2] if len(sys.argv) > 2 else '.'
DEVICE = 'cpu'
torch.set_num_threads(4)

# ---- pull the model, estimator and symbfitrs expression straight from the notebooks --------
# the model, estimator and expression come from the committed notebooks via _neuro (by content);
# this script was first run against an earlier upload of the same notebooks
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from _neuro import ns
gen_gauss, estimator = ns['gen_gauss'], ns['parameter_estimator_bz15']
NUM_STEPS, LR, PATIENCE, SEED = ns['NUM_STEPS'], ns['LR'], ns['PATIENCE'], ns['SEED']

# ---- data: Stringer 2019 gratings (the repo's neuron benchmark) -----------------------------
d = np.load(f'{REPO}/data/external/stringer2019/gratings.npz')
X, Y, ids = d['x'], d['y'], d['cell_ids']
BIN = 8   # trials per angle bin: sorted by angle, pseudo-repeat k takes every 2nd trial of a bin


def binned(r):
    o = np.argsort(X[r]); th, y = X[r][o], Y[r][o]
    n = len(th) // BIN
    th, y = th[:n * BIN].reshape(n, BIN), y[:n * BIN].reshape(n, BIN)
    return th.mean(1), np.stack([y[:, 0::2].mean(1), y[:, 1::2].mean(1)])  # angles, (2, n)


def signal_fraction(R):  # R: (2, n) as in the notebook
    mu, var = R.mean(0), R.var(0, ddof=1); N = R.shape[1]
    S2 = ((mu - mu.mean()) ** 2).mean() - (N - 1) / N**2 * (var / 2).sum()
    V2 = (var / 2).sum() / N
    return float(np.clip(S2 / (S2 + V2 + 1e-12), 0, 1))


def concentration(th, y):
    y = np.clip(y, 0, None); return float(abs((y * np.exp(2j * th)).sum()) / y.sum())


if CELL is None:   # pick a cell passing the notebook screens (SF > 0.6, C > 0.3)
    cands = []
    for r in range(len(X)):
        th, R = binned(r); R = R / np.sqrt((R ** 2).mean(1, keepdims=True))
        sf, c = signal_fraction(R), concentration(X[r], Y[r])
        if sf > 0.6 and c > 0.3: cands.append((r, sf, c))
    print(f'{len(cands)} of {len(X)} cells pass SF>0.6, C>0.3')
    CELL = cands[len(cands) // 2][0]    # the median passing cell, not a cherry-pick
theta_np, R = binned(CELL)
R = R / np.sqrt((R ** 2).mean(1, keepdims=True))
y_train_all, y_test = R[0], R[1]         # fit on repeat 0, test on repeat 1 (as even/odd)
n = len(theta_np)
sf, c = signal_fraction(R), concentration(X[CELL], Y[CELL])
print(f'cell row {CELL} (id {ids[CELL]}): {n} angle bins, SF={sf:.2f}, C={c:.2f}')

idx = np.random.default_rng(SEED).permutation(n); nv = n // 5
val2, val1, train, every = idx[:nv], idx[nv:2 * nv], idx[2 * nv:], np.arange(n)
theta = torch.tensor(theta_np, dtype=torch.float32)
yt = torch.tensor(y_train_all, dtype=torch.float32)


def mse_np(P, bins):   # same model as the Adam notebook evaluates
    with torch.no_grad():
        pred = gen_gauss(theta[bins], torch.as_tensor(P, dtype=torch.float32)[None])[0].numpy()
    return float(np.mean((pred - y_train_all[bins]) ** 2))


def pick_stop(l1, l2):   # the notebook's validation rule, on a per-step loss array (index = step)
    best = int(np.argmin(l1))
    steps = np.clip(best + PATIENCE, 0, len(l1) - 1)
    return int(steps[np.argmin(l2[steps])]), best


# ---- Adam (notebook settings: lr 1e-3, 15 000 steps, loss = 10 * mse) ------------------------
def adam_run(bins, P0, record=None):
    P = torch.tensor(P0, dtype=torch.float32)[None].requires_grad_()
    opt = torch.optim.Adam([P], lr=LR)
    th, y = theta[bins], yt[bins]
    for step in range(NUM_STEPS):
        opt.zero_grad()
        (10 * ((gen_gauss(th, P)[0] - y) ** 2).mean()).backward()
        P.grad.nan_to_num_(0, 0, 0); opt.step()
        if record is not None: record(step + 1, P.detach()[0])
    return P.detach()[0].numpy()


P0_tr = np.asarray(estimator(theta_np[train], y_train_all[train]), dtype=np.float32)
curve = {k: np.full(NUM_STEPS + 1, np.nan) for k in ('train', 'val1', 'val2')}
for k, b in (('train', train), ('val1', val1), ('val2', val2)): curve[k][0] = mse_np(P0_tr, b)


def rec(step, P):
    for k, b in (('train', train), ('val1', val1), ('val2', val2)):
        curve[k][step] = mse_np(P.numpy(), b)


t0 = time.time(); adam_run(train, P0_tr, rec); adam_t = time.time() - t0
adam = {k: v.copy() for k, v in curve.items()}
a_stop, a_best = pick_stop(adam['val1'], adam['val2'])
P0_all = np.asarray(estimator(theta_np, y_train_all), dtype=np.float32)
adam_all = np.full(NUM_STEPS + 1, np.nan); adam_all[0] = mse_np(P0_all, every)
P_at = {}
def rec_all(s, P):
    adam_all[s] = mse_np(P.numpy(), every)
    if s == max(a_stop, 1): P_at['p'] = P.numpy().copy()
adam_run(every, P0_all, rec_all)
print(f'Adam: {adam_t:.1f}s for {NUM_STEPS} steps; val1-best step {a_best}, chosen stop {a_stop}')

# ---- symbfitrs: the new accepted-step trace + max_steps -------------------------------------
import symbfitrs
expr, names = ns['_will_expression']('Gen. Gaussian')
model = symbfitrs.compile(expr)
order = [names.index(p) for p in model.params]        # symbfitrs orders by first appearance
inv = np.argsort(order)                               # model order -> notebook order
BUDGET = 1000.0


def sfit(bins, p0, **kw):
    r = model.fit(theta_np[bins], y_train_all[bins], p0=np.asarray(p0, float)[order], budget_ms=BUDGET, **kw)
    return r, (None if r.trace is None else r.trace[:, inv])


r_tr, tr = sfit(train, P0_tr, trace=True, max_steps=NUM_STEPS)
sym = {k: np.array([mse_np(P, b) for P in tr]) for k, b in (('train', train), ('val1', val1), ('val2', val2))}
s_stop, s_best = pick_stop(sym['val1'], sym['val2'])
r_all, tra = sfit(every, P0_all, trace=True, max_steps=NUM_STEPS)
sym_all = np.array([mse_np(P, every) for P in tra])
print(f'symbfitrs train fit: status={r_tr.status}, {r_tr.steps} accepted steps, {r_tr.time_ms:.1f} ms; '
      f'val1-best step {s_best}, chosen stop {s_stop}')
print(f'symbfitrs all-angle refit: status={r_all.status}, {r_all.steps} accepted steps, {r_all.time_ms:.1f} ms')
r_cap, _ = sfit(every, P0_all, max_steps=s_stop)      # the refit as the protocol runs it
print(f'capped refit: status={r_cap.status}, steps={r_cap.steps}')

# ---- final scores ---------------------------------------------------------------------------
def r2(P):
    with torch.no_grad():
        pred = gen_gauss(theta, torch.as_tensor(P, dtype=torch.float32)[None])[0].numpy()
    return 1 - ((y_test - pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()


res = dict(cell_row=int(CELL), cell_id=int(ids[CELL]), n_bins=int(n), sf=sf, c=c,
           adam=dict(stop=a_stop, best=a_best, secs=adam_t,
                     train=adam['train'].tolist()[::1], val1=adam['val1'].tolist(), val2=adam['val2'].tolist(),
                     all=adam_all.tolist(), start_mse=adam['train'][0]),
           sym=dict(stop=s_stop, best=s_best, steps=int(r_tr.steps), status=r_tr.status, ms=r_tr.time_ms,
                    train=sym['train'].tolist(), val1=sym['val1'].tolist(), val2=sym['val2'].tolist(),
                    all=sym_all.tolist(), refit_steps=int(r_all.steps), refit_status=r_all.status),
           test_r2=dict(start=float(r2(P0_all)),
                        adam_stop=float(r2(P_at['p'])) if 'p' in P_at else float(r2(P0_all)), sym_capped=float(r2(r_cap.params[inv])), sym_full=float(r2(r_all.params[inv]))),
           names=list(names))
json.dump(res, open(f'{OUT}/curves_cell{CELL}.json', 'w'))
np.savez(f'{OUT}/cell{CELL}_data.npz', theta=theta_np, R=R, train=train, val1=val1, val2=val2)
print('saved')
