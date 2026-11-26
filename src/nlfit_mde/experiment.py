"""Experiment configuration (TOML), validated with pydantic.

One experiment = solver/budget settings + a list of datasets. A dataset is either
the built-in synthetic MDE grid or a table of curves on disk plus one or more
models to fit to every curve. See ``configs/`` and ``data/README.md``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import config as defaults
from .models.families import FAMILIES


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StartsSpec(_Strict):
    """How to initialise a model. Named explicit starts and/or random draws."""

    values: dict[str, list[float]] = {}  # e.g. {start1 = [500, 1e-4]}
    bounds: dict[str, tuple[float, float]] = {}  # per-parameter box for random starts
    n_random: int = 0  # starts drawn uniformly from ``bounds``
    near_scale: float | None = None  # adds "near" = reference.params + scale*width*N(0,1)


class ReferenceSpec(_Strict):
    """Known minimum for the success criterion (give at most one)."""

    params: list[float] | None = None  # loss evaluated at these parameters
    sse: float | None = None  # unweighted residual sum of squares (e.g. NIST certified)
    loss: float | None = None  # normalised loss directly

    @model_validator(mode="after")
    def _one(self):
        if sum(v is not None for v in (self.params, self.sse, self.loss)) > 1:
            raise ValueError("give only one of reference.params / sse / loss")
        return self


class ModelSpec(_Strict):
    """An expression model (``expr`` + ``params``) or a built-in family (``family`` + ``p``)."""

    name: str | None = None
    expr: str | None = None
    params: list[str] | None = None
    variables: list[str] | None = None  # default: the dataset's x column names
    linear: list[str] | None = None  # override auto-detected VarPro linear params ([] = none)
    family: str | None = None
    p: int | None = None
    starts: StartsSpec = StartsSpec()
    reference: ReferenceSpec | None = None

    @model_validator(mode="after")
    def _kind(self):
        if (self.expr is None) == (self.family is None):
            raise ValueError("model needs exactly one of `expr` or `family`")
        if self.expr is not None and not self.params:
            raise ValueError(f"expression model {self.expr!r} needs `params`")
        if self.family is not None:
            if self.family not in FAMILIES:
                raise ValueError(f"unknown family {self.family!r}; choose from {sorted(FAMILIES)}")
            if not self.p:
                raise ValueError(f"family {self.family!r} needs `p`")
        return self

    @property
    def label(self) -> str:
        return self.name or self.family or self.expr  # type: ignore[return-value]

    @property
    def param_names(self) -> list[str]:
        if self.params:
            return list(self.params)
        k = (self.p or 0) // 2
        if self.family == "coupled_tanh":
            return [f"t{i}" for i in range(self.p or 0)]
        return [f"a{i}" for i in range(k)] + [f"b{i}" for i in range(k)]


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
    path: Path  # directory with the 27 .dat files (scripts/fetch_benchmarks.py)
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
    path: Path  # gratings.npz from scripts/fetch_benchmarks.py
    max_cells: int | None = None  # None = all cells
    cell_sample: Literal["first", "random"] = "random"
    equations: list[str] | Literal["all"] = "all"
    n_perturbed: int = 1  # extra starts: estimate + perturb_scale * max(|theta|, 0.1) * N(0, 1)
    perturb_scale: float = 0.5
    loss_norm: Literal["mean_square", "variance", "sum_of_squares", "mse"] = "mse"


class TableDataset(_Strict):
    kind: Literal["table"]
    name: str
    path: Path
    format: Literal[
        "auto", "csv", "tsv", "whitespace", "parquet", "feather", "json", "jsonl", "npz"
    ] = "auto"
    read_options: dict[str, Any] = {}  # passed to the pandas reader (comment="#", skiprows=...)
    layout: Literal["long", "wide"] = "long"
    x: list[str]
    y: str | None = None  # long layout: response column
    curve: str | None = None  # long layout: curve-id column (omit for a single curve)
    y_columns: list[str] | None = None  # wide layout: one column per curve (default: all non-x)
    sigma: str | None = None  # long layout: per-point std dev -> residual weights 1/sigma
    max_curves: int | None = None
    curve_sample: Literal["first", "random"] = "first"
    reference: Literal["auto", "given", "best_found", "none"] = "auto"
    loss_norm: Literal["mean_square", "variance", "sum_of_squares", "mse"] = "mean_square"
    models: list[ModelSpec]

    @field_validator("x", mode="before")
    @classmethod
    def _listify(cls, v):
        return [v] if isinstance(v, str) else v

    @model_validator(mode="after")
    def _layout(self):
        if self.layout == "long" and not self.y:
            raise ValueError(f"dataset {self.name!r}: long layout needs `y`")
        if self.layout == "wide" and (self.curve or self.sigma):
            raise ValueError(f"dataset {self.name!r}: `curve`/`sigma` only apply to long layout")
        return self


Dataset = Annotated[
    SyntheticDataset
    | TableDataset
    | NistDataset
    | MghDataset
    | SageFitDataset
    | FitBenchmarkingDataset
    | Stringer2019Dataset,
    Field(discriminator="kind"),
]
PATH_DATASETS = (
    TableDataset,
    NistDataset,
    SageFitDataset,
    FitBenchmarkingDataset,
    Stringer2019Dataset,
)


class ExperimentConfig(_Strict):
    name: str
    seed: int = defaults.SEED
    solvers: list[str] = list(defaults.SOLVER_NAMES)
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
        if bad := set(v) - set(defaults.SOLVER_NAMES):
            raise ValueError(f"unknown solvers {sorted(bad)}; choose from {defaults.SOLVER_NAMES}")
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
