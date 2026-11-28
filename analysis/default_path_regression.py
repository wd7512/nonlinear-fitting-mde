"""Default-path regression check for the feature PRs (stands in for the gate, whose tier files
need full local benchmark runs). Fits the same problems with no bounds and a 1-D p0 under the
installed symbfitrs, saving every parameter vector and the wall time, so two builds can be
compared bit for bit.

usage: <python of a build> analysis/default_path_regression.py OUT.npz
       python analysis/default_path_regression.py --compare A.npz B.npz
Run from the repo root of the analysis branch (needs data/external/stringer2019, analysis/notebooks).
"""
import json, os, sys, time
import numpy as np


def run(out):
    import symbfitrs
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    nb = {n: json.load(open(f'analysis/notebooks/{n}.ipynb')) for n in ('jacob_tuning_fits', 'jacob_tuning_will_fits')}
    cell = lambda b, k: next(''.join(c['source']) for c in nb[b]['cells'] if c['cell_type'] == 'code' and k in ''.join(c['source']))
    ns = {'np': np, 'torch': None}
    w = cell('jacob_tuning_will_fits', 'def _will_expression')
    ns.update(MIN_WIDTH=0.1, MAX_WIDTH=2 * np.pi, P_MIN=0.1, P_MAX=4.0)
    exec(w[w.index('def _will_expression'):w.index('def will_fit')], ns)
    est_src = cell('jacob_tuning_fits', 'def parameter_estimator_bz15')
    exec(est_src.split('MODELS =')[0], ns)          # the estimators only (MODELS needs the torch models)
    expr, names = ns['_will_expression']('Gen. Gaussian')
    d = np.load('data/external/stringer2019/gratings.npz'); X, Y = d['x'], d['y']
    rows = np.array(json.load(open('analysis/robustness_300.json'))['rows'])
    jobs = []
    for r in rows:
        o = np.argsort(X[r]); th, y = X[r][o], Y[r][o]; n = len(th) // 8
        th, y = th[:n * 8].reshape(n, 8).mean(1), y[:n * 8].reshape(n, 8)[:, 0::2].mean(1)
        y = y / np.sqrt((y ** 2).mean())
        p0 = np.asarray(ns['parameter_estimator_bz15'](th, y), float)
        m = symbfitrs.compile(expr)
        jobs.append((m, th, y, p0[[list(names).index(p) for p in m.params]]))
    rng = np.random.default_rng(0)
    xs = np.linspace(0, 5, 50)
    for _ in range(200):
        a, b, c = rng.uniform(0.5, 3), rng.uniform(0.2, 2), rng.uniform(-1, 1)
        jobs.append(('a*exp(-b*x) + c', xs, a * np.exp(-b * xs) + c + 0.01 * rng.standard_normal(50), [1.0, 1.0, 0.0]))
    pool = symbfitrs.Pool(threads=1)
    times = []
    for _ in range(3):
        t0 = time.perf_counter(); res = pool.fit_many(jobs, budget_ms=1000.0, progress=False); times.append(time.perf_counter() - t0)
    np.savez(out, params=np.array([np.pad(r.params, (0, 10 - r.params.size)) for r in res]),
             status=np.array([r.status for r in res]), steps=np.array([r.steps for r in res]), secs=np.array(times))
    print(f'{len(jobs)} fits, best of 3: {min(times):.2f} s; statuses {dict(zip(*np.unique([r.status for r in res], return_counts=True)))}')


def compare(a, b):
    A, B = np.load(a), np.load(b)
    det = (A['status'] != 'deadline') & (B['status'] != 'deadline')
    same = np.all(A['params'] == B['params'], 1)
    print(f'non-deadline fits bit-identical: {same[det].sum()} of {det.sum()}; steps identical: {np.mean(A["steps"][det] == B["steps"][det]):.0%}; '
          f'deadline fits: {(~det).sum()} (timing-dependent, excluded)')
    print(f'wall time, best of 3: {A["secs"].min():.2f} s vs {B["secs"].min():.2f} s ({B["secs"].min() / A["secs"].min() - 1:+.1%})')


if __name__ == '__main__':
    compare(sys.argv[2], sys.argv[3]) if sys.argv[1] == '--compare' else run(sys.argv[1])
