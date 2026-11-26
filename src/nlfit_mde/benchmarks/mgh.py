"""More, Garbow & Hillstrom (1981), "Testing unconstrained optimization software",
ACM TOMS 7(1): the 35 least-squares test problems  f(x) = sum_i r_i(x)^2.

Residuals are written in SymPy (analytic Jacobians via ``VectorResidualModel``).
Variable-dimension problems use the sizes listed in ``SIZES``. ``FSTAR`` holds the
published minimum f* for those sizes; ``tests/test_benchmarks.py`` re-derives each
by local optimisation from x0 (a transcription check).

Standard starts are x0, 10*x0 and 100*x0 (if x0 = 0, only x0 is used).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

S = sp.Rational


@dataclass
class MghProblem:
    key: str  # e.g. "20_watson_n9"
    number: int
    name: str
    n: int
    m: int
    residuals: list
    symbols: list
    x0: np.ndarray
    fstar: float


def _x(n):
    return list(sp.symbols(f"x1:{n + 1}", real=True))


# ---- fixed-size problems -------------------------------------------------------------
def rosenbrock():
    x1, x2 = _x(2)
    return [10 * (x2 - x1**2), 1 - x1], [-1.2, 1.0]


def freudenstein_roth():
    x1, x2 = _x(2)
    return [-13 + x1 + ((5 - x2) * x2 - 2) * x2, -29 + x1 + ((x2 + 1) * x2 - 14) * x2], [0.5, -2.0]


def powell_badly_scaled():
    x1, x2 = _x(2)
    return [10**4 * x1 * x2 - 1, sp.exp(-x1) + sp.exp(-x2) - S(10001, 10000)], [0.0, 1.0]


def brown_badly_scaled():
    x1, x2 = _x(2)
    return [x1 - 10**6, x2 - S(2, 10**6), x1 * x2 - 2], [1.0, 1.0]


def beale():
    x1, x2 = _x(2)
    y = [S(3, 2), S(9, 4), S(21, 8)]
    return [y[i] - x1 * (1 - x2 ** (i + 1)) for i in range(3)], [1.0, 1.0]


def jennrich_sampson(m=10):
    x1, x2 = _x(2)
    return [2 + 2 * i - (sp.exp(i * x1) + sp.exp(i * x2)) for i in range(1, m + 1)], [0.3, 0.4]


def helical_valley():
    x1, x2, x3 = _x(3)
    theta = sp.atan2(x2, x1) / (2 * sp.pi)
    return [10 * (x3 - 10 * theta), 10 * (sp.sqrt(x1**2 + x2**2) - 1), x3], [-1.0, 0.0, 0.0]


def bard():
    x1, x2, x3 = _x(3)
    y = [0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39]
    r = []
    for i in range(1, 16):
        u, v = i, 16 - i
        w = min(u, v)
        r.append(S(str(y[i - 1])) - (x1 + u / (v * x2 + w * x3)))
    return r, [1.0, 1.0, 1.0]


def gaussian():
    x1, x2, x3 = _x(3)
    y = [
        0.0009,
        0.0044,
        0.0175,
        0.0540,
        0.1295,
        0.2420,
        0.3521,
        0.3989,
        0.3521,
        0.2420,
        0.1295,
        0.0540,
        0.0175,
        0.0044,
        0.0009,
    ]
    r = [x1 * sp.exp(-x2 * (S(8 - i, 2) - x3) ** 2 / 2) - S(str(y[i - 1])) for i in range(1, 16)]
    return r, [0.4, 1.0, 0.0]


def meyer():
    x1, x2, x3 = _x(3)
    y = [
        34780,
        28610,
        23650,
        19630,
        16370,
        13720,
        11540,
        9744,
        8261,
        7030,
        6005,
        5147,
        4427,
        3820,
        3307,
        2872,
    ]
    return [x1 * sp.exp(x2 / (45 + 5 * i + x3)) - y[i - 1] for i in range(1, 17)], [
        0.02,
        4000.0,
        250.0,
    ]


def gulf(m=10):
    x1, x2, x3 = _x(3)
    r = []
    for i in range(1, m + 1):
        t = S(i, 100)
        yi = 25 + (-50 * sp.log(t)) ** S(2, 3)
        r.append(sp.exp(-(sp.Abs(yi - x2) ** x3) / x1) - t)
    return r, [5.0, 2.5, 0.15]


def box_3d(m=10):
    x1, x2, x3 = _x(3)
    r = []
    for i in range(1, m + 1):
        t = S(i, 10)
        r.append(sp.exp(-t * x1) - sp.exp(-t * x2) - x3 * (sp.exp(-t) - sp.exp(-10 * t)))
    return r, [0.0, 10.0, 20.0]


def powell_singular():
    x1, x2, x3, x4 = _x(4)
    return [
        x1 + 10 * x2,
        sp.sqrt(5) * (x3 - x4),
        (x2 - 2 * x3) ** 2,
        sp.sqrt(10) * (x1 - x4) ** 2,
    ], [3.0, -1.0, 0.0, 1.0]


def wood():
    x1, x2, x3, x4 = _x(4)
    return [
        10 * (x2 - x1**2),
        1 - x1,
        sp.sqrt(90) * (x4 - x3**2),
        1 - x3,
        sp.sqrt(10) * (x2 + x4 - 2),
        (x2 - x4) / sp.sqrt(10),
    ], [-3.0, -1.0, -3.0, -1.0]


def kowalik_osborne():
    x1, x2, x3, x4 = _x(4)
    y = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
    u = [4, 2, 1, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625]
    r = [
        S(str(y[i]))
        - x1
        * (S(str(u[i])) ** 2 + S(str(u[i])) * x2)
        / (S(str(u[i])) ** 2 + S(str(u[i])) * x3 + x4)
        for i in range(11)
    ]
    return r, [0.25, 0.39, 0.415, 0.39]


def brown_dennis(m=20):
    x1, x2, x3, x4 = _x(4)
    r = []
    for i in range(1, m + 1):
        t = S(i, 5)
        r.append((x1 + t * x2 - sp.exp(t)) ** 2 + (x3 + x4 * sp.sin(t) - sp.cos(t)) ** 2)
    return r, [25.0, 5.0, -5.0, -1.0]


def osborne1():
    x1, x2, x3, x4, x5 = _x(5)
    y = [
        0.844,
        0.908,
        0.932,
        0.936,
        0.925,
        0.908,
        0.881,
        0.850,
        0.818,
        0.784,
        0.751,
        0.718,
        0.685,
        0.658,
        0.628,
        0.603,
        0.580,
        0.558,
        0.538,
        0.522,
        0.506,
        0.490,
        0.478,
        0.467,
        0.457,
        0.448,
        0.438,
        0.431,
        0.424,
        0.420,
        0.414,
        0.411,
        0.406,
    ]
    r = [
        S(str(y[i])) - (x1 + x2 * sp.exp(-10 * i * x4) + x3 * sp.exp(-10 * i * x5))
        for i in range(33)
    ]
    return r, [0.5, 1.5, -1.0, 0.01, 0.02]


def biggs_exp6(m=13):
    x = _x(6)
    r = []
    for i in range(1, m + 1):
        t = S(i, 10)
        y = sp.exp(-t) - 5 * sp.exp(-10 * t) + 3 * sp.exp(-4 * t)
        r.append(x[2] * sp.exp(-t * x[0]) - x[3] * sp.exp(-t * x[1]) + x[5] * sp.exp(-t * x[4]) - y)
    return r, [1.0, 2.0, 1.0, 1.0, 1.0, 1.0]


OSBORNE2_Y = [
    1.366,
    1.191,
    1.112,
    1.013,
    0.991,
    0.885,
    0.831,
    0.847,
    0.786,
    0.725,
    0.746,
    0.679,
    0.608,
    0.655,
    0.616,
    0.606,
    0.602,
    0.626,
    0.651,
    0.724,
    0.649,
    0.649,
    0.694,
    0.644,
    0.624,
    0.661,
    0.612,
    0.558,
    0.533,
    0.495,
    0.500,
    0.423,
    0.395,
    0.375,
    0.372,
    0.391,
    0.396,
    0.405,
    0.428,
    0.429,
    0.523,
    0.562,
    0.607,
    0.653,
    0.672,
    0.708,
    0.633,
    0.668,
    0.645,
    0.632,
    0.591,
    0.559,
    0.597,
    0.625,
    0.739,
    0.710,
    0.729,
    0.720,
    0.636,
    0.581,
    0.428,
    0.292,
    0.162,
    0.098,
    0.054,
]


def osborne2():
    x = _x(11)
    r = []
    for i in range(65):
        t = S(i, 10)
        f = (
            x[0] * sp.exp(-t * x[4])
            + x[1] * sp.exp(-((t - x[8]) ** 2) * x[5])
            + x[2] * sp.exp(-((t - x[9]) ** 2) * x[6])
            + x[3] * sp.exp(-((t - x[10]) ** 2) * x[7])
        )
        r.append(S(str(OSBORNE2_Y[i])) - f)
    return r, [1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 5.5]


# ---- variable-dimension problems ---------------------------------------------------------
def watson(n):
    x = _x(n)
    r = []
    for i in range(1, 30):
        t = S(i, 29)
        s1 = sum((j - 1) * x[j - 1] * t ** (j - 2) for j in range(2, n + 1))
        s2 = sum(x[j - 1] * t ** (j - 1) for j in range(1, n + 1))
        r.append(s1 - s2**2 - 1)
    r += [x[0], x[1] - x[0] ** 2 - 1]
    return r, [0.0] * n


def extended_rosenbrock(n):
    x = _x(n)
    r = []
    for i in range(n // 2):
        r += [10 * (x[2 * i + 1] - x[2 * i] ** 2), 1 - x[2 * i]]
    return r, [-1.2, 1.0] * (n // 2)


def extended_powell(n):
    x = _x(n)
    r = []
    for i in range(n // 4):
        a, b, c, d = x[4 * i : 4 * i + 4]
        r += [a + 10 * b, sp.sqrt(5) * (c - d), (b - 2 * c) ** 2, sp.sqrt(10) * (a - d) ** 2]
    return r, [3.0, -1.0, 0.0, 1.0] * (n // 4)


def penalty1(n):
    x = _x(n)
    a = sp.sqrt(S(1, 10**5))
    return [a * (xi - 1) for xi in x] + [sum(xi**2 for xi in x) - S(1, 4)], [
        float(j) for j in range(1, n + 1)
    ]


def penalty2(n):
    x = _x(n)
    a = sp.sqrt(S(1, 10**5))
    r = [x[0] - S(1, 5)]
    for i in range(2, n + 1):
        y = sp.exp(S(i, 10)) + sp.exp(S(i - 1, 10))
        r.append(a * (sp.exp(x[i - 1] / 10) + sp.exp(x[i - 2] / 10) - y))
    for i in range(n + 1, 2 * n):
        r.append(a * (sp.exp(x[i - n] / 10) - sp.exp(S(-1, 10))))
    r.append(sum((n - j + 1) * x[j - 1] ** 2 for j in range(1, n + 1)) - 1)
    return r, [0.5] * n


def variably_dimensioned(n):
    x = _x(n)
    s = sum(j * (x[j - 1] - 1) for j in range(1, n + 1))
    return [xi - 1 for xi in x] + [s, s**2], [1 - j / n for j in range(1, n + 1)]


def trigonometric(n):
    x = _x(n)
    sc = sum(sp.cos(xj) for xj in x)
    return [n - sc + i * (1 - sp.cos(x[i - 1])) - sp.sin(x[i - 1]) for i in range(1, n + 1)], [
        1 / n
    ] * n


def brown_almost_linear(n):
    x = _x(n)
    s = sum(x)
    return [x[i] + s - (n + 1) for i in range(n - 1)] + [sp.prod(x) - 1], [0.5] * n


def discrete_boundary_value(n):
    x = _x(n)
    h = S(1, n + 1)
    xx = [0, *x, 0]
    r = [
        2 * xx[i] - xx[i - 1] - xx[i + 1] + h**2 * (xx[i] + i * h + 1) ** 3 / 2
        for i in range(1, n + 1)
    ]
    return r, [float(i * h * (i * h - 1)) for i in range(1, n + 1)]


def discrete_integral_equation(n):
    x = _x(n)
    h = S(1, n + 1)
    t = [i * h for i in range(1, n + 1)]
    r = []
    for i in range(n):
        s1 = sum(t[j] * (x[j] + t[j] + 1) ** 3 for j in range(i + 1))
        s2 = sum((1 - t[j]) * (x[j] + t[j] + 1) ** 3 for j in range(i + 1, n))
        r.append(x[i] + h * ((1 - t[i]) * s1 + t[i] * s2) / 2)
    return r, [float(tj * (tj - 1)) for tj in t]


def broyden_tridiagonal(n):
    x = _x(n)
    xx = [0, *x, 0]
    return [(3 - 2 * xx[i]) * xx[i] - xx[i - 1] - 2 * xx[i + 1] + 1 for i in range(1, n + 1)], [
        -1.0
    ] * n


def broyden_banded(n):
    x = _x(n)
    r = []
    for i in range(1, n + 1):
        js = [j for j in range(max(1, i - 5), min(n, i + 1) + 1) if j != i]
        r.append(
            x[i - 1] * (2 + 5 * x[i - 1] ** 2) + 1 - sum(x[j - 1] * (1 + x[j - 1]) for j in js)
        )
    return r, [-1.0] * n


def linear_full_rank(n, m):
    x = _x(n)
    s = sum(x)
    r = [x[i] - S(2, m) * s - 1 for i in range(n)] + [-S(2, m) * s - 1 for _ in range(n, m)]
    return r, [1.0] * n


def linear_rank1(n, m):
    x = _x(n)
    s = sum(j * x[j - 1] for j in range(1, n + 1))
    return [i * s - 1 for i in range(1, m + 1)], [1.0] * n


def linear_rank1_zero(n, m):
    x = _x(n)
    s = sum(j * x[j - 1] for j in range(2, n))
    return [S(-1)] + [(i - 1) * s - 1 for i in range(2, m)] + [S(-1)], [1.0] * n


def chebyquad(n):
    x = _x(n)
    z = sp.Symbol("z")
    r = []
    for i in range(1, n + 1):
        Ti = sp.chebyshevt(i, 2 * z - 1)
        integral = S(-1, i * i - 1) if i % 2 == 0 else 0
        r.append(sum(Ti.subs(z, xj) for xj in x) / n - integral)
    return r, [j / (n + 1) for j in range(1, n + 1)]


# number, name, builder, f* (published, for the sizes used here)
N10 = 10
REGISTRY = [
    (1, "rosenbrock", rosenbrock, 0.0),
    (2, "freudenstein_roth", freudenstein_roth, 0.0),
    (3, "powell_badly_scaled", powell_badly_scaled, 0.0),
    (4, "brown_badly_scaled", brown_badly_scaled, 0.0),
    (5, "beale", beale, 0.0),
    (6, "jennrich_sampson", jennrich_sampson, 124.362),
    (7, "helical_valley", helical_valley, 0.0),
    (8, "bard", bard, 8.21487e-3),
    (9, "gaussian", gaussian, 1.12793e-8),
    (10, "meyer", meyer, 87.9458),
    (11, "gulf", gulf, 0.0),
    (12, "box_3d", box_3d, 0.0),
    (13, "powell_singular", powell_singular, 0.0),
    (14, "wood", wood, 0.0),
    (15, "kowalik_osborne", kowalik_osborne, 3.07505e-4),
    (16, "brown_dennis", brown_dennis, 85822.2),
    (17, "osborne1", osborne1, 5.46489e-5),
    (18, "biggs_exp6", biggs_exp6, 0.0),
    (19, "osborne2", osborne2, 4.01377e-2),
    (20, "watson_n6", lambda: watson(6), 2.28767e-3),
    (20, "watson_n9", lambda: watson(9), 1.39976e-6),
    (21, "extended_rosenbrock_n10", lambda: extended_rosenbrock(N10), 0.0),
    (22, "extended_powell_n12", lambda: extended_powell(12), 0.0),
    (23, "penalty1_n10", lambda: penalty1(N10), 7.08765e-5),
    (24, "penalty2_n10", lambda: penalty2(N10), 2.93660e-4),
    (25, "variably_dimensioned_n10", lambda: variably_dimensioned(N10), 0.0),
    (26, "trigonometric_n10", lambda: trigonometric(N10), 0.0),
    (27, "brown_almost_linear_n10", lambda: brown_almost_linear(N10), 0.0),
    (28, "discrete_boundary_value_n10", lambda: discrete_boundary_value(N10), 0.0),
    (29, "discrete_integral_equation_n10", lambda: discrete_integral_equation(N10), 0.0),
    (30, "broyden_tridiagonal_n10", lambda: broyden_tridiagonal(N10), 0.0),
    (31, "broyden_banded_n10", lambda: broyden_banded(N10), 0.0),
    (32, "linear_full_rank_n10_m20", lambda: linear_full_rank(N10, 20), 10.0),
    (33, "linear_rank1_n10_m20", lambda: linear_rank1(N10, 20), 20 * 19 / (2 * 41)),
    (
        34,
        "linear_rank1_zero_n10_m20",
        lambda: linear_rank1_zero(N10, 20),
        (400 + 60 - 6) / (2 * 37),
    ),
    (35, "chebyquad_n8", lambda: chebyquad(8), 3.51687e-3),
]


def load(names: list[str] | None = None) -> list[MghProblem]:
    out = []
    for num, name, build, fstar in REGISTRY:
        key = f"{num:02d}_{name}"
        if names and name not in names and key not in names and str(num) not in names:
            continue
        res, x0 = build()
        syms = sorted(
            set().union(*[sp.sympify(r).free_symbols for r in res]), key=lambda s: int(s.name[1:])
        )
        n = len(x0)
        syms = _x(n) if len(syms) != n else syms  # keep x1..xn even if some never appear
        out.append(
            MghProblem(
                key,
                num,
                name,
                n,
                len(res),
                [sp.sympify(r) for r in res],
                syms,
                np.asarray(x0, float),
                float(fstar),
            )
        )
    return out


def build(spec, rng=None):
    """Problems for a ``kind = "mgh"`` dataset: y = 0, loss = sum of squares, R = min(f*, best found)."""
    from ..models import VectorResidualModel
    from ..problem import Problem

    names = None if spec.problems == "all" else list(spec.problems)
    out = []
    for mp in load(names):
        model = VectorResidualModel(mp.key, mp.residuals, mp.symbols, center=mp.x0)
        scales = [1.0] if not np.any(mp.x0) else spec.scales
        starts = {}
        for s in scales:
            x0 = mp.x0 * s
            with np.errstate(all="ignore"):
                f0 = float(np.sum(model.predict(x0) ** 2))
            if np.isfinite(f0):  # 100 x0 overflows for a few problems: skip that start
                starts[f"{s:g}x0"] = x0
        out.append(
            Problem(
                pid=f"{spec.name}/{mp.key}",
                dataset=spec.name,
                model_name=mp.key,
                curve=mp.key,
                model=model,
                y=np.zeros(mp.m),
                starts=starts,
                param_scale=np.maximum(np.abs(mp.x0), 1.0),
                ref_loss=mp.fstar,
                ref_mode="given_or_best",
                loss_norm="sum_of_squares",
                tags={"size": "n<=4" if mp.n <= 4 else "n>4"},
            )
        )
    return out
