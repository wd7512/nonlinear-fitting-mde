"""Learning trajectories: baseline LM vs the proposed fixes (and Adam), per accepted step and per unit of time.
Variants (Rust prototype, 300 Stringer neurons, same raw estimator start):
  baseline            current solver on the notebook's clipped model
  step cap 0.1        same, each step limited to 0.1 per parameter
  bounds              active-set bounds on the UNCLIPPED model (bounds replace the clips)
  bounds + cap 0.3    both
  Adam                15,000 steps, lr 1e-3, batched over the 300 neurons (time amortised per neuron)
usage: PYTHONPATH=. python analysis/trajectories.py OUT.npz
"""
import sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, symbfitrs
from _neuro import *

OUT = sys.argv[1]; torch.set_num_threads(4)
d = np.load('analysis/diagnosis.npz', allow_pickle=True); ab = d['ab']; L_a_final = d['L_a']
c = np.load('analysis/bounds_step_cap.npz', allow_pickle=True); best = c['best']
u = np.load('analysis/bounds_unclipped_rust.npz', allow_pickle=True); best = np.minimum(best, u['L'].min(0))
N = len(ROWS); cells = [trials(r) for r in ROWS]
TH = np.stack([x[0] for x in cells]); TRAIN = np.stack([x[1][0].mean(1) for x in cells]); TWO_PI = 2 * np.pi; nb = list(NAMES)
P0 = np.array([np.asarray(estimator(TH[i], TRAIN[i]), float) for i in range(N)])
LO = np.array([0, 0, 0, 0, .1, .1, .1, .1, .1, .1]); HI = np.array([TWO_PI, np.inf, np.inf, np.inf] + [TWO_PI] * 4 + [4., 4.])
BOUNDS = {'b': (0, None), 'a1': (0, None), 'a2': (0, None), **{k: (0.1, TWO_PI) for k in ('l1', 'r1', 'l2', 'r2')}, 'p1': (0.1, 4.0), 'p2': (0.1, 4.0)}
e = EXPR; W = '6.283185307179586'
e = e.replace(f'min(max(theta, 0), {W})', 'theta')
for n in ('l1', 'r1', 'l2', 'r2'): e = e.replace(f'min(max({n}, 0.1), {W})', n)
for n in ('p1', 'p2'): e = e.replace(f'min(max({n}, 0.1), 4.0)', n)
i1, i2 = e.index(' + max(a1, 0)*'), e.index(' + max(a2, 0)*')
FREE = symbfitrs.compile(f'b + a1*{e[i1 + len(" + max(a1, 0)*"):i2]} + a2*{e[i2 + len(" + max(a2, 0)*"):]}')
order_f = [nb.index(p) for p in FREE.params]; inv_f = np.argsort(order_f)
th64, y64 = torch.tensor(TH, dtype=torch.float64), torch.tensor(TRAIN, dtype=torch.float64)

def traj_loss(i, Pnb):            # train MSE of the notebook's clipped model along a (steps, 10) parameter trace
    with torch.no_grad(): return ((gen_gauss(th64[i], torch.tensor(Pnb, dtype=torch.float64)) - y64[i]) ** 2).mean(1).numpy()
def dead_count(Pnb): return ((Pnb[:, 1:] < LO[1:] - 1e-12) | (Pnb[:, 1:] > HI[1:] + 1e-12)).sum(1)   # the nine bounded parameters

variants = {'baseline (current solver)': (MODEL, ORDER, None, None), 'step cap 0.1': (MODEL, ORDER, None, 0.1),
            'bounds replace clips': (FREE, order_f, BOUNDS, None), 'bounds + step cap 0.3': (FREE, order_f, BOUNDS, 0.3)}
rec = {}
for lab, (model, order, bnd, cap) in variants.items():
    inv = np.argsort(order)
    def one(i):
        r = model.fit(TH[i], TRAIN[i], p0=P0[i][order], budget_ms=3000.0, trace=True, bounds=bnd, max_step=cap)
        Pt = r.trace[:, inv]; return traj_loss(i, Pt), dead_count(Pt), r.history_ms, r.history_loss
    t = time.time()
    with ThreadPoolExecutor(4) as ex: out = list(ex.map(one, range(N)))
    rec[lab] = out; print(f'{lab:28s} {time.time() - t:5.1f} s; median accepted steps {np.median([len(o[0]) - 1 for o in out]):.0f}', flush=True)

# Adam, batched, recording per-neuron loss and the number of dead parameters at every step
STEPS = 15000; th32 = torch.tensor(TH, dtype=torch.float32); y32 = torch.tensor(TRAIN, dtype=torch.float32)
Pa = torch.tensor(P0.astype(np.float32)).requires_grad_(); opt = torch.optim.Adam([Pa], lr=1e-3)
adam_loss = np.empty((STEPS + 1, N), np.float32); adam_dead = np.empty((STEPS + 1, N), np.uint8)
lo_t, hi_t = torch.tensor(LO[1:], dtype=torch.float32), torch.tensor(HI[1:], dtype=torch.float32)
t = time.time()
for s in range(STEPS):
    opt.zero_grad(); lc = ((gen_gauss(th32, Pa).squeeze(0) - y32) ** 2).mean(1); (10 * lc.mean()).backward()
    adam_loss[s] = lc.detach().numpy(); adam_dead[s] = ((Pa.detach()[:, 1:] < lo_t - 1e-12) | (Pa.detach()[:, 1:] > hi_t + 1e-12)).sum(1).numpy()
    Pa.grad.nan_to_num_(0, 0, 0); opt.step()
with torch.no_grad(): adam_loss[STEPS] = ((gen_gauss(th32, Pa).squeeze(0) - y32) ** 2).mean(1).numpy()
adam_secs = time.time() - t; print(f'Adam {adam_secs:.0f} s for {STEPS} steps x {N} neurons', flush=True)

# aggregate curves on common grids
sgrid = np.unique(np.r_[0, np.round(np.logspace(0, np.log10(STEPS), 140)).astype(int)])
tgrid = np.logspace(-1, np.log10(3000.0), 140)                              # ms
def at_steps(l):   return np.array([l[min(s, len(l) - 1)] for s in sgrid])
def at_times(ms, loss):
    out = np.empty(len(tgrid)); j = np.searchsorted(ms, tgrid, side='right') - 1; out = np.where(j >= 0, np.asarray(loss)[np.clip(j, 0, None)], np.inf); return out
agg = {}
for lab, o in rec.items():
    agg[lab] = dict(steps=np.array([at_steps(x[0]) for x in o]), time=np.array([at_times(x[2], x[3]) for x in o]))
agg['Adam (batched; time amortised per neuron)'] = dict(steps=adam_loss[sgrid].T, time=None)
amort = adam_secs / STEPS / N * 1e3                                          # ms per step per neuron
agg['Adam (batched; time amortised per neuron)']['time'] = np.array([[adam_loss[min(int(t / amort), STEPS), i] for t in tgrid] for i in range(N)])
idx = np.where(ab)[0]
np.savez_compressed(OUT, sgrid=sgrid, tgrid=tgrid, best=best, ab=ab, labels=list(agg), steps=np.array([agg[k]['steps'] for k in agg], dtype=np.float32),
         time=np.array([agg[k]['time'] for k in agg], dtype=np.float32), adam_amortised_ms_per_step=amort, rows=ROWS, gap_idx=idx,
         adam_loss_gap=adam_loss[:, idx], adam_dead_gap=adam_dead[:, idx],
         **{f'loss::{k}': np.array([np.pad(v[i][0], (0, 6000 - len(v[i][0])), constant_values=np.nan)[:6000] for i in idx], dtype=np.float32) for k, v in rec.items()},
         **{f'dead::{k}': np.array([np.pad(v[i][1].astype(float), (0, 6000 - len(v[i][1])), constant_values=np.nan)[:6000] for i in idx], dtype=np.float32) for k, v in rec.items()})
print('saved', OUT)
