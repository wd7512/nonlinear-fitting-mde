"""Experiment configuration (TOML), validated with pydantic.

One experiment = baseline/budget settings + a list of suites. See ``bench/configs/``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import config as defaults
from .baselines import BASELINES


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SyntheticDataset(_Strict):
    kind: Literal["synthetic"]
    name: str = "synthetic"
    families: list[str] = list(defaults.FAMILIES)
    param_counts: list[int] = list(defaults.PARAM_COUNTS)
    near_scale: float = defaults.NEAR_SCALE
    n_curves: int = 1  # independent curves (true parameters) per family x p
    loss_norm: Literal["mean_square", "variance", "sum_of_squares", "mse"] = "mean_square"


class NistDataset(_Strict):
    """NIST StRD nonlinear regression (.dat files with certified values)."""

    kind: Literal["nist"]
    name: str = "nist"
    path: Path  # directory with the 27 .dat files (python -m bench.fetch)
    problems: list[str] | Literal["all"] = "all"
    reference: Literal["certified", "best_found"] = "certified"
    loss_norm: Literal["mean_square", "variance", "sum_of_squares", "mse"] = "mean_square"


class MghDataset(_Strict):
    """More, Garbow & Hillstrom (1981) unconstrained least-squares test problems."""

    kind: Literal["mgh"]
    name: str = "mgh"
    problems: list[str] | Literal["all"] = "all"
    scales: list[float] = [1.0, 10.0, 100.0]  # starts: x0, 10 x0, 100 x0 (MGH convention)


class SageFitDataset(_Strict):
    """SAGE-Fit Structure-Progress Bank: 20 problems x 10 candidate equations."""

    kind: Literal["sagefit_spb"]
    name: str = "spb"
    path: Path  # root of the CAS-CLab/SAGE-Fit checkout
    split: Literal["train", "test", "ood_test"] = "train"
    max_problems: int | None = None
    random_starts: int = 0  # extra uniform starts in [-2, 2] besides the cold start (ones)
    loss_norm: Literal["mean_square", "variance", "sum_of_squares", "mse"] = "variance"


class FitBenchmarkingDataset(_Strict):
    """FitBenchmarking problem definition files whose Mantid functions are reimplemented here."""

    kind: Literal["fitbenchmarking"]
    name: str = "fitbenchmarking"
    path: Path  # fitbenchmarking/examples/benchmark_problems
    sets: list[str]  # sub-folders, e.g. ["Neutron", "Muon", "simple_tests"]
    use_errors: bool = True  # weighted least squares with the data's error column
    loss_norm: Literal["mean_square", "variance", "sum_of_squares", "mse"] = "mean_square"


class Stringer2019Dataset(_Strict):
    """Stringer et al. (2019) V1 gratings cells x candidate tuning equations (reference = best
    found)."""

    kind: Literal["stringer2019_gratings"]
    name: str = "stringer2019"
    path: Path  # gratings.npz from python -m bench.fetch
    max_cells: int | None = None  # None = all cells
    cell_sample: Literal["first", "random"] = "random"
    equations: list[str] | Literal["all"] = "all"
    n_perturbed: int = 1  # extra starts: estimate + perturb_scale * max(|theta|, 0.1) * N(0, 1)
    perturb_scale: float = 0.5
    loss_norm: Literal["mean_square", "variance", "sum_of_squares", "mse"] = "mse"


Dataset = Annotated[
    SyntheticDataset
    | NistDataset
    | MghDataset
    | SageFitDataset
    | FitBenchmarkingDataset
    | Stringer2019Dataset,
    Field(discriminator="kind"),
]
PATH_DATASETS = (
    NistDataset,
    SageFitDataset,
    FitBenchmarkingDataset,
    Stringer2019Dataset,
)


class ExperimentConfig(_Strict):
    name: str
    seed: int = defaults.SEED
    solvers: list[str] = list(BASELINES)  # registered baselines
    budgets_ms: list[float] = list(defaults.BUDGETS_MS)
    targets: list[float] = list(defaults.TARGETS)
    worker_deadline_s: float = defaults.WORKER_DEADLINE_S
    watchdog_s: float = defaults.WATCHDOG_S
    # Success at a checkpoint (L = best loss so far, R = reference, L0 = initial loss):
    #   absolute   L - R <= target
    #   relative   L - R <= target * max(R, relative_floor)
    #   reduction  L - R <= target * (L0 - R)        (More & Wild 2009 convergence test)
    success: Literal["absolute", "relative", "reduction"] = "absolute"
    relative_floor: float = 1e-20
    full_history: bool = False  # save every finite trial instead of first + improvements
    datasets: list[Dataset]

    @field_validator("solvers")
    @classmethod
    def _known(cls, v):
        if bad := set(v) - set(BASELINES):
            raise ValueError(f"unknown baselines {sorted(bad)}; choose from {list(BASELINES)}")
        return v

    @property
    def run_deadline_ms(self) -> float:
        return float(max(self.budgets_ms))


def _resolve(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    for root in (Path.cwd(), base):
        if (root / path).exists():
            return (root / path).resolve()
    return (Path.cwd() / path).resolve()


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    cfg = ExperimentConfig.model_validate(raw)
    datasets = [
        d.model_copy(update={"path": _resolve(d.path, path.parent)})
        if isinstance(d, PATH_DATASETS)
        else d
        for d in cfg.datasets
    ]
    return cfg.model_copy(update={"datasets": datasets})
