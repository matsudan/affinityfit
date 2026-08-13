"""The sample code does not rot as the API changes.

`examples/plot_fit.py` is not part of the distribution, but it exists to be read and copied, so it has to keep
working. The tests call it for real to confirm that.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from affinityfit import fit, hill, langmuir

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load_sample():
    """Load examples/plot_fit.py as a module."""
    path = EXAMPLES / "plot_fit.py"
    spec = importlib.util.spec_from_file_location("sample_plot_fit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sample():
    return load_sample()


def titration(kd=10.0, bmax=1.0, baseline=0.02, noise=0.02, seed=3):
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 11)])
    signal = langmuir(conc, kd, bmax, baseline) + np.random.default_rng(seed).normal(0, noise, conc.size)
    return conc, signal


def test_sample_writes_a_png(sample, tmp_path):
    conc, signal = titration()
    res = fit(conc, signal, unit="nM")
    out = sample.plot_fit(conc, signal, res, tmp_path / "fit.png")
    assert out.exists() and out.stat().st_size > 1000


def test_sample_handles_a_decreasing_signal(sample, tmp_path):
    """A negative amplitude, as in fluorescence quenching, can also be plotted."""
    conc, _ = titration()
    signal = langmuir(conc, 10.0, -0.8, 1.0)
    res = fit(conc, signal, unit="nM")
    out = sample.plot_fit(conc, signal, res, tmp_path / "quench.png")
    assert out.exists() and out.stat().st_size > 1000


def test_sample_handles_a_model_with_extra_parameters(sample, tmp_path):
    conc, _ = titration()
    signal = hill(conc, 10.0, 1.0, 0.0, 2.0)
    res = fit(conc, signal, model=hill)
    out = sample.plot_fit(conc, signal, res, tmp_path / "hill.png")
    assert out.exists()


def test_sample_rejects_data_without_a_positive_concentration(sample, tmp_path):
    conc = np.zeros(6)
    signal = np.zeros(6)
    res = fit(np.array([0.0, 1.0, 3.0, 10.0, 30.0, 100.0]), np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9]))
    with pytest.raises(ValueError, match="No positive concentration"):
        sample.plot_fit(conc, signal, res, tmp_path / "bad.png")


def test_sample_main_runs_end_to_end(sample, tmp_path, capsys):
    out = tmp_path / "out.png"
    status = sample.main([str(EXAMPLES / "titration_good.csv"), "--unit", "nM", "--out", str(out)])
    assert status == 0
    assert out.exists()
    printed = capsys.readouterr().out
    assert "Kd" in printed and "saved:" in printed


def test_sample_main_accepts_every_model(sample, tmp_path):
    from affinityfit import MODELS

    for name in MODELS:
        out = tmp_path / f"{name}.png"
        assert sample.main([str(EXAMPLES / "titration_good.csv"), "--model", name, "--out", str(out)]) == 0
        assert out.exists()


# ------------------------------- 例データが例として妥当であること


def test_example_files_declare_that_they_are_synthetic():
    """論文由来と誤解されないよう、由来をファイル内に書いておく。"""
    for name in ("titration_good.csv", "titration_unsaturated.csv"):
        text = (EXAMPLES / name).read_text(encoding="utf-8")
        assert "not from any publication" in text, name
        assert "make_examples.py" in text, name
        assert "seed=" in text, name


def test_good_example_is_consistent_with_the_model_it_illustrates(sample):
    """「良いデータ」が協同性ありと判定されるようでは例にならない。"""
    from affinityfit import fit, hill

    conc, signal = sample.read_csv(EXAMPLES / "titration_good.csv")
    res = fit(conc, signal, ci="profile")
    assert res.warnings == (), res.warnings
    assert res.r_squared > 0.999

    cooperative = fit(conc, signal, model=hill, ci="profile")
    assert cooperative.intervals["n"].contains(1.0)
    assert res.aicc < cooperative.aicc  # 余分なパラメータは支持されない


def test_unsaturated_example_still_shows_an_undetermined_limit(sample):
    from affinityfit import fit

    conc, signal = sample.read_csv(EXAMPLES / "titration_unsaturated.csv")
    res = fit(conc, signal, ci="profile")
    assert res.r_squared > 0.99  # 当てはまりは良く見える
    assert res.intervals["kd"].upper is None  # それでも上限は決まらない
    assert "not_saturated" in {diagnostic.code for diagnostic in res.warnings}


def test_examples_can_be_regenerated_reproducibly(tmp_path, monkeypatch):
    """スクリプトを 2 回走らせても同じ内容になること。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sample_make_examples", EXAMPLES.parent / "scripts" / "make_examples.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    original = (EXAMPLES / "titration_good.csv").read_text(encoding="utf-8")
    monkeypatch.setitem(module.GOOD, "path", tmp_path / "again.csv")
    module.write(module.GOOD)
    assert (tmp_path / "again.csv").read_text(encoding="utf-8") == original
