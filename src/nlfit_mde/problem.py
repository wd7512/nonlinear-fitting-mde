"""A fitting problem: one model, one curve, named starting points, optional reference."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import Model

LOSS_NORMS = ("mean_square", "variance", "sum_of_squares", "mse")


def loss_denominator(y_eff: np.ndarray, norm: str) -> float:
    """Denominator of the loss: loss = sum(r^2) / m / denominator.

    mean_square     mean(y^2)   (MDE default: normalised MSE)
    variance        var(y)      (NMSE as used by SRBench / SAGE-Fit)
    sum_of_squares  1/m         (loss = plain sum of squares; residual-only problems, MGH)
    mse             1           (plain mean squared error, in the data's units)
    """
    if norm == "mse":
        return 1.0
    if norm == "sum_of_squares":
        return 1.0 / y_eff.size
    if norm == "variance":
        return max(float(np.var(y_eff)), 1e-30)
    return max(float(np.mean(y_eff * y_eff)), 1e-12)


@dataclass
class Problem:
    pid: str
    dataset: str
    model_name: str
    curve: str
    model: Model
    y: np.ndarray
    starts: dict[str, np.ndarray]
    param_scale: np.ndarray  # typical |theta| per parameter (Adam / L-BFGS scaling)
    weights: np.ndarray | None = None
    ref_loss: float | None = None  # known minimum loss, if any
    ref_mode: str = "none"  # zero | given | best_found | given_or_best | none
    ref_params: np.ndarray | None = None  # parameters known to reach ref_loss (e.g. certified)
    loss_norm: str = "mean_square"
    tags: dict = field(default_factory=dict)

    @property
    def p(self) -> int:
        return self.model.p

    @property
    def m(self) -> int:
        return self.model.m

    @property
    def y_eff(self) -> np.ndarray:
        return self.y if self.weights is None else self.weights * self.y

    @property
    def denom(self) -> float:
        return loss_denominator(self.y_eff, self.loss_norm)

    def loss_at(self, theta: np.ndarray) -> float:
        w = 1.0 if self.weights is None else self.weights
        with np.errstate(all="ignore"):
            r = w * (self.model.predict(theta) - self.y)
            return float(np.dot(r, r)) / self.m / self.denom

    def loss_from_sse(self, sse: float) -> float:
        """Loss for an (unweighted or weighted, matching the problem) residual sum of squares."""
        return sse / self.m / self.denom


def default_param_scale(starts: dict[str, np.ndarray], bounds: tuple | None = None) -> np.ndarray:
    if bounds is not None:
        lo, hi = bounds
        return np.maximum(np.abs(lo), np.abs(hi))
    s = np.max(np.abs(np.vstack(list(starts.values()))), axis=0)
    return np.where(s > 0, s, 1.0)
