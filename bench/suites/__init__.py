"""Turn an experiment config into a flat list of ``Problem``s."""

from __future__ import annotations

import numpy as np

from ..experiment import (
    ExperimentConfig,
    FitBenchmarkingDataset,
    MghDataset,
    NistDataset,
    SageFitDataset,
    Stringer2019Dataset,
    SyntheticDataset,
)
from ..problem import Problem
from . import fitbenchmarking, mgh, nist, sagefit, stringer2019, synthetic


def build_problems(cfg: ExperimentConfig, rng: np.random.Generator) -> list[Problem]:
    builders = {
        SyntheticDataset: synthetic.build,
        NistDataset: nist.build,
        MghDataset: mgh.build,
        SageFitDataset: sagefit.build,
        FitBenchmarkingDataset: fitbenchmarking.build,
        Stringer2019Dataset: stringer2019.build,
    }
    problems: list[Problem] = []
    for ds in cfg.datasets:
        problems += builders[type(ds)](ds, rng)
    if len({p.pid for p in problems}) != len(problems):
        raise ValueError("duplicate problem ids: give datasets unique names")
    return problems


__all__ = ["build_problems"]
