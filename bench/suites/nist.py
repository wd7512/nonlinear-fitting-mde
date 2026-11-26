"""NIST Statistical Reference Datasets (StRD), nonlinear regression: 27 problems.

https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml  (public domain)

Each .dat file holds the data, two official starting points, certified parameter
values (computed in 128-bit precision) and the certified residual sum of squares.
Model expressions are transcribed below; ``tests/test_benchmarks.py`` checks every
one against the certified RSS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DIFFICULTY = {
    "lower": [
        "Misra1a",
        "Chwirut2",
        "Chwirut1",
        "Lanczos3",
        "Gauss1",
        "Gauss2",
        "DanWood",
        "Misra1b",
    ],
    "average": [
        "Kirby2",
        "Hahn1",
        "Nelson",
        "MGH17",
        "Lanczos1",
        "Lanczos2",
        "Gauss3",
        "Misra1c",
        "Misra1d",
        "Roszman1",
        "ENSO",
    ],
    "higher": ["MGH09", "Thurber", "BoxBOD", "Rat42", "MGH10", "Eckerle4", "Rat43", "Bennett5"],
}
NAMES = [n for v in DIFFICULTY.values() for n in v]
URL = "https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/{name}.dat"

_LANCZOS = "b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)"
_GAUSS = "b1*exp(-b2*x) + b3*exp(-(x-b4)**2/b5**2) + b6*exp(-(x-b7)**2/b8**2)"
_RATIONAL3 = "(b1 + b2*x + b3*x**2 + b4*x**3) / (1 + b5*x + b6*x**2 + b7*x**3)"
_P = "2*pi*x"

MODELS = {
    "Misra1a": "b1*(1 - exp(-b2*x))",
    "Chwirut2": "exp(-b1*x) / (b2 + b3*x)",
    "Chwirut1": "exp(-b1*x) / (b2 + b3*x)",
    "Lanczos3": _LANCZOS,
    "Gauss1": _GAUSS,
    "Gauss2": _GAUSS,
    "DanWood": "b1*x**b2",
    "Misra1b": "b1*(1 - (1 + b2*x/2)**(-2))",
    "Kirby2": "(b1 + b2*x + b3*x**2) / (1 + b4*x + b5*x**2)",
    "Hahn1": _RATIONAL3,
    "Nelson": "b1 - b2*x1*exp(-b3*x2)",  # response is log(y)
    "MGH17": "b1 + b2*exp(-x*b4) + b3*exp(-x*b5)",
    "Lanczos1": _LANCZOS,
    "Lanczos2": _LANCZOS,
    "Gauss3": _GAUSS,
    "Misra1c": "b1*(1 - (1 + 2*b2*x)**(-1/2))",
    "Misra1d": "b1*b2*x / (1 + b2*x)",
    "Roszman1": "b1 - b2*x - atan(b3/(x - b4))/pi",
    "ENSO": f"b1 + b2*cos({_P}/12) + b3*sin({_P}/12) + b5*cos({_P}/b4) + b6*sin({_P}/b4)"
    f" + b8*cos({_P}/b7) + b9*sin({_P}/b7)",
    "MGH09": "b1*(x**2 + x*b2) / (x**2 + x*b3 + b4)",
    "Thurber": _RATIONAL3,
    "BoxBOD": "b1*(1 - exp(-b2*x))",
    "Rat42": "b1 / (1 + exp(b2 - b3*x))",
    "MGH10": "b1*exp(b2/(x + b3))",
    "Eckerle4": "(b1/b2) * exp(-(x - b3)**2 / (2*b2**2))",
    "Rat43": "b1 / (1 + exp(b2 - b3*x))**(1/b4)",
    "Bennett5": "b1*(b2 + x)**(-1/b3)",
}
LOG_RESPONSE = {"Nelson"}

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


@dataclass
class NistProblem:
    name: str
    difficulty: str
    expr: str
    params: list[str]
    variables: list[str]
    x: np.ndarray  # (m, d)
    y: np.ndarray
    starts: dict[str, np.ndarray]
    certified: np.ndarray
    certified_sd: np.ndarray
    certified_rss: float


def parse_dat(path: Path) -> NistProblem:
    name = path.stem
    lines = path.read_text(errors="replace").splitlines()
    pat = re.compile(rf"^\s*b(\d+)\s*=\s*({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})")
    rows = sorted(
        (int(m[1]), [float(m[k]) for k in range(2, 6)]) for line in lines if (m := pat.match(line))
    )
    vals = np.array([r for _, r in rows])
    rss = next(
        float(re.search(_NUM, line.split(":", 1)[1])[0])
        for line in lines
        if line.strip().startswith("Residual Sum of Squares")
    )
    data_hdr = max(i for i, line in enumerate(lines) if line.strip().startswith("Data:"))
    cols = lines[data_hdr].split(":", 1)[1].split()
    data = np.array(
        [[float(v) for v in line.split()] for line in lines[data_hdr + 1 :] if line.strip()]
    )
    variables = [c for c in cols if c != "y"]
    y = data[:, cols.index("y")]
    if name in LOG_RESPONSE:
        y = np.log(y)
    x = data[:, [cols.index(v) for v in variables]]
    difficulty = next((d for d, names in DIFFICULTY.items() if name in names), "unknown")
    return NistProblem(
        name=name,
        difficulty=difficulty,
        expr=MODELS[name],
        params=[f"b{i}" for i, _ in rows],
        variables=variables,
        x=x,
        y=y,
        starts={"start1": vals[:, 0], "start2": vals[:, 1]},
        certified=vals[:, 2],
        certified_sd=vals[:, 3],
        certified_rss=rss,
    )


def load(directory: Path, names: list[str] | None = None) -> list[NistProblem]:
    names = names or NAMES
    missing = [n for n in names if not (directory / f"{n}.dat").exists()]
    if missing:
        raise FileNotFoundError(
            f"NIST files missing in {directory}: {missing}; run python -m bench.fetch"
        )
    return [parse_dat(directory / f"{n}.dat") for n in names]


def lre(estimate: np.ndarray, certified: np.ndarray, digits: float = 11.0) -> np.ndarray:
    """Log relative error per parameter (number of correct significant digits), capped at ``digits``."""
    est, cert = np.asarray(estimate, float), np.asarray(certified, float)
    with np.errstate(all="ignore"):
        rel = np.abs(est - cert) / np.where(cert != 0, np.abs(cert), 1.0)
        out = -np.log10(rel)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=digits, neginf=0.0), 0.0, digits)


def build(spec, rng=None):
    """Problems for a ``kind = "nist"`` dataset.

    Reference (``certified``): the smaller of the loss at the certified parameters
    (evaluated in float64; equals the certified RSS to ~1e-11 except Lanczos1, whose
    1.4e-25 is below what 11-digit parameters reproduce) and the best loss found.
    """
    from ..models import SymbolicModel
    from ..problem import Problem

    names = None if spec.problems == "all" else list(spec.problems)
    out = []
    for nb in load(spec.path, names):
        model = SymbolicModel(nb.name, nb.expr, nb.params, nb.variables, nb.x)
        model.set_linear(model.detect_linear(center=nb.starts["start1"]))
        pb = Problem(
            pid=f"{spec.name}/{nb.name}",
            dataset=spec.name,
            model_name=nb.name,
            curve=nb.name,
            model=model,
            y=nb.y,
            starts=nb.starts,
            ref_params=nb.certified,
            loss_norm=spec.loss_norm,
            tags={"difficulty": nb.difficulty},
        )
        if spec.reference == "certified":
            pb.ref_loss = pb.loss_at(nb.certified)
            pb.ref_mode = "given_or_best"
        else:
            pb.ref_mode = "best_found"
        out.append(pb)
    return out
