"""(1) Noise ceiling of held-out R^2 with 4 trials per bin. (2) Does the 100 ms budget change the
bootstrap exponent spread? Re-run 20 bootstraps on 100 neurons at 100 ms vs 2000 ms."""
import numpy as np
from _neuro import *

cells = [trials(r) for r in ROWS]
ceil = []
for th, tr in cells:
    test = tr[1]; m = test.mean(1)
    noise = (test.var(1, ddof=1) / test.shape[1]).mean()      # variance of the 4-trial bin means
    ceil.append(1 - noise / ((m - m.mean()) ** 2).mean())     # R^2 of the true curve vs the noisy test half
ceil = np.array(ceil)
print(f'(1) noise ceiling of held-out R2 (perfect model, 4 trials/bin): median {np.median(ceil):.3f}, IQR {np.percentile(ceil,25):.3f}-{np.percentile(ceil,75):.3f}')
print('    the notebook generalised Gaussian reaches median 0.657 (all 10 candidate equations: 0.642-0.657)')

rng = np.random.default_rng(5); sub = list(range(0, len(ROWS), 3)); B = 20
est = lambda th, y: np.asarray(estimator(th, y), dtype=np.float32)
curves = {b: [] for b in range(B)}
for b in range(B):
    for ci in sub:
        th, tr = cells[ci]; idx = rng.integers(0, 4, (len(th), 4)); curves[b].append(np.take_along_axis(tr[0], idx, 1).mean(1))
out = {}
for budget in (100.0, 2000.0):
    sds, dl, steps = [], 0, []
    pm = np.zeros((len(sub), B))
    for b in range(B):
        P, rs = fit_many([cells[ci][0] for ci in sub], curves[b], np.array([est(cells[ci][0], c) for ci, c in zip(sub, curves[b])]), budget_ms=budget)
        pm[:, b] = clipped_p(P).min(1); dl += sum(r.status == 'deadline' for r in rs); steps += [r.steps for r in rs]
    out[budget] = pm
    print(f'(2) budget {budget:6.0f} ms: {dl} of {B * len(sub)} fits hit the budget; median bootstrap SD of min(p) {np.median(pm.std(1)):.2f}, '
          f'median 95% width {np.median(np.percentile(pm, 97.5, 1) - np.percentile(pm, 2.5, 1)):.2f}')
