"""Turn an experiment config into a flat list of ``Problem``s."""

from __future__ import annotations

import numpy as np

from ..benchmarks import fitbenchmarking, mgh, nist, sagefit, stringer2019
from ..experiment import (
    ExperimentConfig,
    FitBenchmarkingDataset,
    MghDataset,
    ModelSpec,
    NistDataset,
    SageFitDataset,
    Stringer2019Dataset,
    SyntheticDataset,
    TableDataset,
)
from ..models import Model, SymbolicModel, make_family
from ..problem import Problem, default_param_scale
from . import synthetic
from .starts import bounds_arrays, make_starts
from .tables import Curve, iter_curves


def _build_model(ms: ModelSpec, ds: TableDataset, curve: Curve) -> Model:
    if ms.family:
        if curve.x.shape[1] != 1:
            raise ValueError(f"family {ms.family!r} needs a single x column")
        return make_family(ms.family, curve.x[:, 0], ms.p, name=ms.label)  # type: ignore[arg-type]
    return SymbolicModel(ms.label, ms.expr, ms.params, ms.variables or ds.x, curve.x, ms.linear)  # type: ignore[arg-type]


def _table_problems(ds: TableDataset, rng: np.random.Generator) -> list[Problem]:
    problems = []
    for curve in iter_curves(ds, rng):
        weights = None if curve.sigma is None else 1.0 / curve.sigma
        for ms in ds.models:
            model = _build_model(ms, ds, curve)
            names = model.param_names
            ref = ms.reference
            ref_params = np.asarray(ref.params, float) if ref and ref.params is not None else None
            if ref_params is not None and ref_params.size != model.p:
                raise ValueError(
                    f"{ms.label}: reference.params has {ref_params.size} values, need {model.p}"
                )
            starts = make_starts(ms.starts, names, rng, ref_params)
            pb = Problem(
                pid=f"{ds.name}/{ms.label}/{curve.curve_id}",
                dataset=ds.name,
                model_name=ms.label,
                curve=curve.curve_id,
                model=model,
                y=curve.y,
                starts=starts,
                param_scale=default_param_scale(starts, bounds_arrays(ms.starts, names)),
                weights=weights,
                ref_params=ref_params,
                loss_norm=ds.loss_norm,
            )
            mode = ds.reference
            if mode in ("auto", "given") and ref is not None:
                if ref.params is not None:
                    pb.ref_loss = pb.loss_at(ref_params)
                elif ref.sse is not None:
                    pb.ref_loss = pb.loss_from_sse(ref.sse)
                else:
                    pb.ref_loss = ref.loss
                pb.ref_mode = "given"
            elif mode == "given":
                raise ValueError(
                    f"dataset {ds.name!r} has reference='given' but model {ms.label!r} has none"
                )
            elif mode in ("auto", "best_found"):
                pb.ref_mode = "best_found"  # filled post hoc by the report
            if model.m < model.p:
                raise ValueError(f"{pb.pid}: m={model.m} observations < p={model.p} parameters")
            problems.append(pb)
    if not problems:
        raise ValueError(f"dataset {ds.name!r} produced no problems")
    return problems


def build_problems(cfg: ExperimentConfig, rng: np.random.Generator) -> list[Problem]:
    problems: list[Problem] = []
    builders = {
        SyntheticDataset: synthetic.build,
        TableDataset: _table_problems,
        NistDataset: nist.build,
        MghDataset: mgh.build,
        SageFitDataset: sagefit.build,
        FitBenchmarkingDataset: fitbenchmarking.build,
        Stringer2019Dataset: stringer2019.build,
    }
    for ds in cfg.datasets:
        problems += builders[type(ds)](ds, rng)
    if len({p.pid for p in problems}) != len(problems):
        raise ValueError("duplicate problem ids: give datasets/models unique names")
    return problems


__all__ = ["build_problems"]
