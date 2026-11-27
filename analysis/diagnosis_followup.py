"""Follow-up to adam_vs_lm_diagnosis.py: is the difference parameters locked out by the model's clips?"""
import numpy as np
from _neuro import *
import torch
d = np.load('analysis/diagnosis.npz', allow_pickle=True)
P_s, P_a, P0, ab, L_s, L_a = d['P_s'], d['P_a'], d['P0'], d['ab'], d['L_s'], d['L_a']
fd, steps = d['first_dead'], d['sym_steps']
NAMES10 = ['theta', 'b', 'a1', 'a2', 'l1', 'r1', 'l2', 'r2', 'p1', 'p2']
LO = np.array([0, 0, 0, 0, .1, .1, .1, .1, .1, .1]); HI = np.array([2 * np.pi, np.inf, np.inf, np.inf] + [2 * np.pi] * 4 + [4., 4.])
dead = lambda P: (np.asarray(P) < LO) | (np.asarray(P) > HI)
N = len(P_s); cells = [trials(r) for r in ROWS]; TH = np.stack([c[0] for c in cells]); TRAIN = np.stack([c[1][0].mean(1) for c in cells])
gg = lambda th, P: gen_gauss(th, P).squeeze(0)
th64, y64 = torch.tensor(TH, dtype=torch.float64), torch.tensor(TRAIN, dtype=torch.float64)
mse64 = lambda P: ((gg(th64, torch.tensor(P, dtype=torch.float64)) - y64) ** 2).mean(1).detach().numpy()

# (a) is the estimator's starting point already partly outside the clips?
D0 = dead(P0)
print('(a) parameters outside their clip range AT THE ESTIMATOR START (zero gradient for both optimisers from step 0):')
print(f'    any dead: Adam-better {D0[ab].any(1).mean():.0%} (n={ab.sum()}), others {D0[~ab].any(1).mean():.0%};  ' +
      ', '.join(f'{n} {D0[:, j].mean():.0%}' for j, n in enumerate(NAMES10) if D0[:, j].any()))
print(f'    number of dead parameters at the start: Adam-better mean {D0[ab].sum(1).mean():.2f}, others {D0[~ab].sum(1).mean():.2f}')

# (b) when does LM first push a parameter into a dead region?
m = fd >= 0
print('\n(b) LM: accepted step at which a parameter first leaves its clip range (-1 = never; step 0 = already dead at the start):')
print(f'    never: {np.mean(fd < 0):.0%};  at the start (step 0): {np.mean(fd == 0):.0%};  after at least one LM step: {np.mean(fd > 0):.0%}')
print(f'    among neurons where it happens after the start: median step {np.median(fd[fd > 0]):.0f} of {np.median(steps[fd > 0]):.0f} total; Adam-better {np.median(fd[ab & (fd > 0)]) if (ab & (fd > 0)).any() else float("nan"):.0f}')
print(f'    entered a dead region AFTER the start: Adam-better {np.mean(fd[ab] > 0):.0%} vs others {np.mean(fd[~ab] > 0):.0%}')

# (c) in the Adam-better neurons, which parameters are dead for LM but alive for Adam, and vice versa
Ds, Da = dead(P_s), dead(P_a)
print('\n(c) Adam-better neurons: parameters dead in the LM solution but alive in Adam\'s (and the reverse):')
print('    LM-dead/Adam-alive: ' + ', '.join(f'{n} {(Ds & ~Da)[ab][:, j].sum()}' for j, n in enumerate(NAMES10)) + f'  (neurons with any: {(Ds & ~Da)[ab].any(1).sum()} of {ab.sum()})')
print('    Adam-dead/LM-alive: ' + ', '.join(f'{n} {(Da & ~Ds)[ab][:, j].sum()}' for j, n in enumerate(NAMES10)) + f'  (neurons with any: {(Da & ~Ds)[ab].any(1).sum()} of {ab.sum()})')

# (d) revival test: put every dead parameter just inside its range and re-run LM from there
eps = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
inner_lo, inner_hi = LO + eps, np.where(np.isfinite(HI), HI - eps, np.inf)
P_rev = np.where(Ds, np.clip(P_s, inner_lo, inner_hi), P_s)
P_rev2, rs = fit_many(TH, TRAIN, P_rev, budget_ms=1000.0)
L_rev = mse64(P_rev2)
print('\n(d) revival test: LM restarted from its own solution with every dead parameter moved just inside its range:')
print(f'    new loss vs old: lower by >0.1% in {np.mean(L_rev < L_s * .999):.0%} of all neurons, >1% in {np.mean(L_rev < L_s * .99):.0%}; '
      f'Adam-better neurons: >0.1% in {np.mean((L_rev < L_s * .999)[ab]):.0%}, >1% in {np.mean((L_rev < L_s * .99)[ab]):.0%}')
print(f'    Adam-better neurons that then reach Adam\'s level (within 1%): {np.mean((L_rev <= L_a * 1.01)[ab]):.0%} (before: {np.mean((L_s <= L_a * 1.01)[ab]):.0%})')
print(f'    all neurons: LM-from-revived is within 0.1% of or below Adam in {np.mean(L_rev <= L_a * 1.001):.0%} (single estimator start: {np.mean(L_s <= L_a * 1.001):.0%}); Adam still >1% lower in {np.mean(L_a < L_rev * .99):.0%} (before {np.mean(L_a < L_s * .99):.0%})')
print(f'    median loss change {np.median(L_rev / L_s - 1):+.1e}; statuses {dict(zip(*np.unique([r.status for r in rs], return_counts=True)))}')
np.savez('analysis/diagnosis_followup.npz', L_rev=L_rev, P_rev=P_rev2, D0=D0)
