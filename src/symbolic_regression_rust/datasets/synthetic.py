"""Built-in synthetic benchmark fixtures: families x parameter counts, near/far starts.

Inputs live on x in [0, 1]; true parameters are drawn from bounded boxes with
moderate rates / frequencies so fixture generation cannot overflow. Nonlinear
parameters of the mixtures are stratified across their range so terms are distinct
(high-order mixtures are still poorly identifiable, by design). Noise-free, so the
reference minimum is zero.
"""

from __future__ import annotations

import numpy as np

from .. import config
from ..models import make_family
from ..problem import Problem

EXP_A = (0.5, 1.5)
EXP_B = (0.3, 6.0)  # decay rates; stratified on a log grid
SIN_A = (0.5, 1.5)  # magnitude; sign is random
SIN_B = (1.0, 4.0 * np.pi)  # angular frequencies, up to two cycles on [0, 1]
COUPLED = (-1.5, 1.5)


def _stratified(rng, lo, hi, k, log):
    u = (np.arange(k) + rng.uniform(size=k)) / k
    if log:
        return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
    return lo + u * (hi - lo)


OFFSET = (-0.5, 0.5)  # constant term of odd-p mixtures


def _box(family: str, p: int) -> tuple[np.ndarray, np.ndarray]:
    k, odd = p // 2, p % 2
    if family == "coupled_tanh":
        return np.full(p, COUPLED[0]), np.full(p, COUPLED[1])
    a, b = (EXP_A, EXP_B) if family == "exp_mixture" else ((-SIN_A[1], SIN_A[1]), SIN_B)
    lo = np.r_[np.full(k, a[0]), np.full(k, b[0]), [OFFSET[0]] * odd]
    hi = np.r_[np.full(k, a[1]), np.full(k, b[1]), [OFFSET[1]] * odd]
    return lo, hi


def _true_params(family, p, rng):
    k, odd = p // 2, p % 2
    if family == "coupled_tanh":
        return rng.uniform(*COUPLED, size=p)
    if family == "exp_mixture":
        theta = np.r_[rng.uniform(*EXP_A, size=k), _stratified(rng, *EXP_B, k, log=True)]
    else:
        a = rng.choice([-1.0, 1.0], size=k) * rng.uniform(*SIN_A, size=k)
        theta = np.r_[a, _stratified(rng, *SIN_B, k, log=False)]
    return np.r_[theta, rng.uniform(*OFFSET, size=odd)]


def _starts(theta_true, lo, hi, family, rng, near_scale):
    near = theta_true + near_scale * (hi - lo) * rng.standard_normal(theta_true.size)
    if family == "exp_mixture":  # far: independent draw from the box (log-uniform rates)
        k = theta_true.size // 2
        a, b, c = slice(0, k), slice(k, 2 * k), slice(2 * k, None)
        far = np.r_[
            rng.uniform(lo[a], hi[a]),
            np.exp(rng.uniform(np.log(lo[b]), np.log(hi[b]))),
            rng.uniform(lo[c], hi[c]),  # offset (odd p only; draws nothing for even p)
        ]
    else:
        far = rng.uniform(lo, hi)
    return {"near": near, "far": far}


def make_problem(
    family: str,
    p: int,
    rng: np.random.Generator,
    dataset: str = "synthetic",
    near_scale: float = config.NEAR_SCALE,
    curve: int | None = None,
    loss_norm: str = "mean_square",
) -> Problem:
    x = np.linspace(0.0, 1.0, max(64, 4 * p))
    lo, hi = _box(family, p)
    model = make_family(family, x, p)
    theta_true = _true_params(family, p, rng)
    y = model.predict(theta_true)  # FP64, noise-free
    if not np.all(np.isfinite(y)):
        raise RuntimeError(f"non-finite fixture for {family} p={p}")
    high_order = family != "coupled_tanh" and p >= config.HIGH_ORDER_MIXTURE_P
    suffix = "" if curve is None else f"/{curve}"
    return Problem(
        pid=f"{dataset}/{family}_p{p}{suffix}",
        dataset=dataset,
        model_name=family,
        curve=f"{family}_p{p}{suffix}",  # one model per curve: no candidate ranking
        model=model,
        y=y,
        starts=_starts(theta_true, lo, hi, family, rng, near_scale),
        param_scale=np.maximum(np.abs(lo), np.abs(hi)),
        ref_loss=0.0,
        ref_mode="zero",
        ref_params=theta_true,
        loss_norm=loss_norm,
        tags={"family": family, "regime": "high-order mixture" if high_order else "well-posed"},
    )


def build(spec, rng: np.random.Generator) -> list[Problem]:
    out = []
    for f in spec.families:
        for p in spec.param_counts:
            for c in range(spec.n_curves):
                curve = None if spec.n_curves == 1 else c
                out.append(
                    make_problem(f, p, rng, spec.name, spec.near_scale, curve, spec.loss_norm)
                )
    return out


def make_warmup_fixture() -> Problem:
    """Tiny separate fixture (own RNG) used only to warm up solver code paths."""
    return make_problem("exp_mixture", 4, np.random.default_rng(config.SEED + 1), dataset="warmup")
