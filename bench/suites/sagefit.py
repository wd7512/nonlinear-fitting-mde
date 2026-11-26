"""SAGE-Fit Structure-Progress Bank (SPB): 20 dynamical-system problems x 10 LLM-proposed
candidate equations ``dP/dt = equation(t, P, params)`` (arXiv:2605.23272,
https://github.com/CAS-CLab/SAGE-Fit).

Protocol as in the paper's baseline: cold start params = ones, NMSE loss
``mean((f - y)^2) / var(y)`` (``loss_norm = "variance"``), per-candidate reference
S_ref = best NMSE any evaluator found; a candidate is *lost* if NMSE > 3 S_ref.
Candidates are traced to SymPy when possible (analytic Jacobian, VarPro structure);
the rest keep their code with a finite-difference Jacobian.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BANK = Path("Structure_Progress_Bank/equations/high_potential_per_problem.json")
TARGET_COLUMN = "dP_dt"


def build(spec, rng: np.random.Generator):
    from ..models import python_function_model
    from ..problem import Problem

    root = Path(spec.path)
    bank = json.loads((root / BANK).read_text())
    if spec.max_problems:
        bank = bank[: spec.max_problems]
    out = []
    for prob in bank:
        eq = prob["equation_id"]
        data = pd.read_csv(root / "data" / eq / f"{spec.split}.csv")
        xcols = [c for c in data.columns if c != TARGET_COLUMN]
        x, y = data[xcols].to_numpy(float), data[TARGET_COLUMN].to_numpy(float)
        for k, ev in enumerate(prob["events"]):
            model = python_function_model(f"{eq}/c{k}", ev["function"], xcols, x)
            starts = {"ones": np.ones(model.p)}
            for j in range(spec.random_starts):
                starts[f"rand{j}"] = rng.uniform(-2.0, 2.0, model.p)
            out.append(
                Problem(
                    pid=f"{spec.name}/{eq}/c{k}",
                    dataset=spec.name,
                    model_name=f"c{k}",
                    curve=eq,
                    model=model,
                    y=y,
                    starts=starts,
                    ref_mode="best_found",
                    loss_norm=spec.loss_norm,
                    tags={
                        "jacobian": model.jacobian_kind,
                        "candidate": f"c{k}",
                        "paper_nmse": float(ev.get("new_best_nmse", "nan")),
                    },
                )
            )
    return out
