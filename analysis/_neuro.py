"""Shared setup for the neuron-fitting analyses: the jacob_tuning notebooks' model, estimator and
symbfitrs expression (extracted from the notebooks, not retyped) and the Stringer 2019 trials."""
import glob, json
import numpy as np
import torch
import symbfitrs

UP = glob.glob('/root/.claude/uploads/*/')[0]
# the two notebooks as uploaded (newest upload of each name wins); cells are found by content,
# not by position, so an edit to a notebook does not silently shift what is executed
_nb = lambda name: json.load(open(sorted(glob.glob(UP + f'*-{name}.ipynb'), key=lambda f: __import__('os').path.getmtime(f))[-1]))
_adam, _will = _nb('jacob_tuning_fits'), _nb('jacob_tuning_will_fits')
def cell(nb, needle):
    hits = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code' and needle in ''.join(c['source'])]
    assert len(hits) == 1, f'{len(hits)} cells contain {needle!r}'
    return hits[0]
ns = {'np': np, 'torch': torch}
exec(cell(_adam, 'def gen_gauss'), ns); exec(cell(_adam, 'def parameter_estimator_bz15'), ns)
exec(cell(_adam, 'NUM_STEPS, LR').split('def mse')[0], ns)       # NUM_STEPS, LR, PATIENCE, SEED
_w = cell(_will, 'def _will_expression'); exec(_w[_w.index('def _will_expression'):_w.index('def will_fit')], ns)
_pen = cell(_will, 'def penalised_fit')
exec(_pen[:_pen.index('def penalised_fit')], ns)                  # PENALTY, PENALTY_BUDGET_MS, PENALTY_SCALES
PENALISED_FIT_SRC = _pen[_pen.index('def penalised_fit'):_pen.index('def fit(')]
gen_gauss, estimator = ns['gen_gauss'], ns['parameter_estimator_bz15']
P_MIN, P_MAX = ns['P_MIN'], ns['P_MAX']
EXPR, NAMES = ns['_will_expression']('Gen. Gaussian')
MODEL = symbfitrs.compile(EXPR)
ORDER = [NAMES.index(p) for p in MODEL.params]    # symbfitrs orders parameters by first appearance
INV = np.argsort(ORDER)                           # symbfitrs order -> notebook order

_d = np.load('data/external/stringer2019/gratings.npz')
X, Y, IDS = _d['x'], _d['y'], _d['cell_ids']
ROWS = np.array(json.load(open('analysis/robustness_300.json'))['rows'])   # the 300 random neurons
BIN = 8


def trials(r):
    """One neuron: bin-centre angles (n,), raw trials per bin (n, 8) -- columns alternate the two
    pseudo-repeats -- and the RMS scale of each pseudo-repeat's binned mean (as the notebook normalises)."""
    o = np.argsort(X[r]); th, y = X[r][o], Y[r][o]; n = len(th) // BIN
    th, y = th[:n * BIN].reshape(n, BIN), y[:n * BIN].reshape(n, BIN)
    rep = [y[:, 0::2], y[:, 1::2]]
    scale = [np.sqrt((t.mean(1) ** 2).mean()) for t in rep]
    return th.mean(1), [t / s for t, s in zip(rep, scale)]      # angles, 2 x (n, 4) normalised trials


def model_np(P, theta):
    """gen_gauss at one cell's angles for a stack of parameter vectors P (k, 10) -> (k, bins)."""
    with torch.no_grad():
        return gen_gauss(torch.as_tensor(theta, dtype=torch.float32), torch.as_tensor(P, dtype=torch.float32)).numpy()


def clipped_p(P):
    """The two exponents as the model uses them, (k, 2)."""
    return np.clip(np.asarray(P)[:, 8:10], P_MIN, P_MAX)


def fit_many(thetas, curves, p0s, budget_ms=100.0, threads=None, **kw):
    """Fit the notebook's generalised Gaussian to many curves; returns parameter arrays in the
    notebook's order and the FitResults."""
    jobs = [(MODEL, th, y, p0[ORDER]) for th, y, p0 in zip(thetas, curves, p0s)]
    res = symbfitrs.Pool(threads=threads).fit_many(jobs, budget_ms=budget_ms, progress=False, **kw)
    P = np.array([r.params[INV] if r.params.size else np.full(10, np.nan) for r in res])
    return P, res


def r2(y, pred):
    return 1 - ((y - pred) ** 2).sum(-1) / ((y - y.mean(-1, keepdims=True)) ** 2).sum(-1)
