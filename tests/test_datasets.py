import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from nlfit_mde.datasets import build_problems
from nlfit_mde.experiment import ExperimentConfig, load_config


def _cfg(**ds):
    return ExperimentConfig.model_validate({"name": "t", "datasets": [ds]})


MODEL = {"expr": "a*exp(-b*x)", "params": ["a", "b"], "starts": {"values": {"s": [1.0, 1.0]}}}


def test_long_layout_with_curves_and_sigma(tmp_path):
    x = np.linspace(0, 1, 10)
    df = pd.concat(
        [pd.DataFrame({"id": k, "x": x, "y": (k + 1) * np.exp(-x), "s": 0.1}) for k in range(3)]
    )
    df.to_csv(tmp_path / "d.csv", index=False)
    cfg = _cfg(
        kind="table",
        name="d",
        path=str(tmp_path / "d.csv"),
        x="x",
        y="y",
        curve="id",
        sigma="s",
        models=[MODEL],
        max_curves=2,
    )
    pbs = build_problems(cfg, np.random.default_rng(0))
    assert [p.curve for p in pbs] == ["0", "1"]
    assert np.allclose(pbs[0].weights, 10.0)
    assert pbs[1].loss_at(np.array([2.0, 1.0])) < 1e-30


def test_wide_layout_npz(tmp_path):
    x = np.linspace(0, 1, 8)
    np.savez(tmp_path / "w.npz", x=x, y1=np.exp(-x), y2=2 * np.exp(-x))
    cfg = _cfg(
        kind="table", name="w", path=str(tmp_path / "w.npz"), layout="wide", x="x", models=[MODEL]
    )
    assert [p.curve for p in build_problems(cfg, np.random.default_rng(0))] == ["y1", "y2"]


def test_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    x = np.linspace(0, 1, 8)
    pd.DataFrame({"x": x, "y": np.exp(-x)}).to_parquet(tmp_path / "p.parquet")
    cfg = _cfg(
        kind="table", name="p", path=str(tmp_path / "p.parquet"), x="x", y="y", models=[MODEL]
    )
    assert len(build_problems(cfg, np.random.default_rng(0))) == 1


def test_missing_column_message(tmp_path):
    pd.DataFrame({"t": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}).to_csv(
        tmp_path / "d.csv", index=False
    )
    cfg = _cfg(kind="table", name="d", path=str(tmp_path / "d.csv"), x="x", y="y", models=[MODEL])
    with pytest.raises(ValueError, match=r"columns \['x'\] not found"):
        build_problems(cfg, np.random.default_rng(0))


def test_config_validation():
    with pytest.raises(ValidationError, match="exactly one of `expr` or `family`"):
        _cfg(kind="table", name="d", path="x.csv", x="x", y="y", models=[{"params": ["a"]}])
    with pytest.raises(ValidationError):
        _cfg(kind="table", name="d", path="x.csv", x="x", y="y", models=[MODEL], typo_field=1)


def test_misra1a_reference_from_certified_sse(root):
    cfg = load_config(root / "configs/examples/nist_misra1a.toml")
    (pb,) = build_problems(cfg, np.random.default_rng(0))
    certified = np.array([2.3894212918e02, 5.5015643181e-04])
    assert pb.ref_mode == "given"
    assert pb.loss_at(certified) == pytest.approx(pb.ref_loss, rel=1e-8)


@pytest.mark.parametrize(
    "name", ["mde_synthetic.toml", "examples/nist_misra1a.toml", "examples/damped_curves.toml"]
)
def test_shipped_configs_build(root, name):
    cfg = load_config(root / "configs" / name)
    assert build_problems(cfg, np.random.default_rng(cfg.seed))
