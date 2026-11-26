import pytest

from bench import cli


@pytest.mark.parametrize("config", ["quick"])
def test_run_writes_a_report_and_plot(root, tmp_path, monkeypatch, config):
    monkeypatch.chdir(root)  # the worker subprocess imports bench from here
    cli.main(["run", config, "--out", str(tmp_path)])
    summary = (tmp_path / "summary.md").read_text()
    assert "scipy_trf" in summary and "scipy_lm" in summary
    assert (tmp_path / "success_vs_time.png").stat().st_size > 0
