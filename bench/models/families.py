"""Equation families (hand-written kernels) used by the synthetic suite.

theta layout for mixtures: [a_1..a_K, b_1..b_K], conditionally linear in a.
"""

from __future__ import annotations

import numpy as np

from .base import Model


class MixtureModel(Model):
    """f(x) = sum_k a_k phi(b_k x) [+ c].  Even p: K = p/2 terms. Odd p: K = (p-1)/2 terms plus a
    constant offset c, so any p >= 2 is valid (e.g. p = 99). Linear parameters: a (and c)."""

    def __init__(self, name: str, x: np.ndarray, p: int):
        if p < 2:
            raise ValueError(f"{name} needs p >= 2, got {p}")
        k = p // 2
        self.k, self.offset = k, bool(p % 2)
        names = (
            [f"a{i}" for i in range(k)]
            + [f"b{i}" for i in range(k)]
            + (["c"] if self.offset else [])
        )
        linear = list(range(k)) + ([p - 1] if self.offset else [])
        super().__init__(name, np.ravel(x), names, linear_idx=linear)

    def basis(self, beta):
        raise NotImplementedError

    def basis_and_deriv(self, beta):
        raise NotImplementedError

    def _parts(self, theta):
        k = self.k
        return theta[:k], theta[k : 2 * k], (theta[2 * k] if self.offset else 0.0)

    def predict(self, theta):
        a, b, c = self._parts(theta)
        return self.basis(b) @ a + c

    def jacobian(self, theta):
        a, b, _ = self._parts(theta)
        phi, dphi = self.basis_and_deriv(b)
        cols = [phi, dphi * a] + ([np.ones((self.m, 1))] if self.offset else [])
        return np.hstack(cols)


class ExpMixture(MixtureModel):
    """f(x) = sum_k a_k exp(-b_k x)."""

    def basis(self, beta):
        return np.exp(-np.outer(self.x, beta))

    def basis_and_deriv(self, beta):
        phi = self.basis(beta)
        return phi, -self.x[:, None] * phi


class SinMixture(MixtureModel):
    """f(x) = sum_k a_k sin(b_k x)."""

    def basis(self, beta):
        return np.sin(np.outer(self.x, beta))

    def basis_and_deriv(self, beta):
        arg = np.outer(self.x, beta)
        return np.sin(arg), self.x[:, None] * np.cos(arg)


class CoupledTanh(Model):
    """f(x) = tanh(sum_k theta_k cos(k pi x) / sqrt(p)); non-separable."""

    def __init__(self, name: str, x: np.ndarray, p: int):
        super().__init__(name, np.ravel(x), [f"t{i}" for i in range(p)])
        self.design = np.cos(np.pi * np.outer(self.x, np.arange(p))) / np.sqrt(p)

    def predict(self, theta):
        return np.tanh(self.design @ theta)

    def jacobian(self, theta):
        f = np.tanh(self.design @ theta)
        return (1.0 - f * f)[:, None] * self.design


FAMILIES: dict[str, type[Model]] = {
    "exp_mixture": ExpMixture,
    "sin_mixture": SinMixture,
    "coupled_tanh": CoupledTanh,
}


def make_family(family: str, x: np.ndarray, p: int, name: str | None = None) -> Model:
    try:
        cls = FAMILIES[family]
    except KeyError:
        raise ValueError(f"unknown family {family!r}; choose from {sorted(FAMILIES)}") from None
    return cls(name or family, x, p)
