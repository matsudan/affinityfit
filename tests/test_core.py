"""Whether known parameters are recovered, and whether the diagnostics fire as intended."""

from __future__ import annotations

import numpy as np
import pytest

from bindfit import diagnose, fit, langmuir, load_csv


def make_data(kd=10.0, bmax=1.0, baseline=0.02, cmax_factor=30.0, n=12, noise=0.0, seed=0):
    """Synthesise titration data from known parameters."""
    conc = np.concatenate([[0.0], np.logspace(np.log10(kd / 30), np.log10(kd * cmax_factor), n - 1)])
    signal = langmuir(conc, kd, bmax, baseline)
    if noise:
        signal = signal + np.random.default_rng(seed).normal(0.0, noise * bmax, size=signal.shape)
    return conc, signal


def test_recovers_known_kd_without_noise():
    conc, signal = make_data(kd=10.0, bmax=1.0, baseline=0.02)
    res = fit(conc, signal)
    assert res.params["kd"] == pytest.approx(10.0, rel=1e-6)
    assert res.params["bmax"] == pytest.approx(1.0, rel=1e-6)
    assert res.params["baseline"] == pytest.approx(0.02, rel=1e-6)
    assert res.r_squared > 0.9999
    assert res.warnings == ()


def test_recovers_known_kd_with_noise():
    conc, signal = make_data(kd=47.0, bmax=250.0, baseline=5.0, n=12, noise=0.02, seed=42)
    res = fit(conc, signal)
    assert res.params["kd"] == pytest.approx(47.0, rel=0.15)
    assert res.params["kd"] - res.ci95["kd"] < 47.0 < res.params["kd"] + res.ci95["kd"]


def test_recovers_nanomolar_kd():
    """Whether the fit stays numerically stable at a nanomolar Kd expressed in molar units."""
    conc, signal = make_data(kd=4.7e-8, bmax=1.0, baseline=0.0, n=10)
    res = fit(conc, signal)
    assert res.params["kd"] == pytest.approx(4.7e-8, rel=1e-5)


def test_confidence_interval_widens_with_noise():
    conc, clean = make_data(kd=10.0, n=12, noise=0.0)
    _, noisy = make_data(kd=10.0, n=12, noise=0.05, seed=1)
    ci_clean = fit(conc, clean).ci95["kd"]
    ci_noisy = fit(conc, noisy).ci95["kd"]
    assert ci_noisy > ci_clean


def test_location_property_follows_the_model():
    conc, signal = make_data(kd=10.0)
    res = fit(conc, signal)
    assert res.location == res.params["kd"]


def test_fixed_parameter_is_reported_and_held():
    conc, signal = make_data(kd=10.0, baseline=0.0)
    res = fit(conc, signal, fixed={"baseline": 0.0})
    assert res.params["baseline"] == 0.0
    assert res.ci95["baseline"] == 0.0
    assert res.fixed == {"baseline": 0.0}
    assert "(fixed)" in res.report()
    assert res.params["kd"] == pytest.approx(10.0, rel=1e-4)


def test_rejects_more_parameters_than_points():
    conc = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="data point"):
        fit(conc, np.array([0.1, 0.2]))


# --------------------------------------------------------------------- diagnostics


def test_warns_when_saturation_not_reached():
    """Highest concentration below 3 times Kd -> warning that saturation was not reached."""
    conc, signal = make_data(kd=100.0, cmax_factor=2.0, n=8)
    res = fit(conc, signal)
    assert any("飽和に達しておらず" in w for w in res.warnings)


def test_warns_when_max_conc_below_10x_kd():
    conc, signal = make_data(kd=100.0, cmax_factor=5.0, n=8)
    res = fit(conc, signal)
    assert any("10 倍未満" in w for w in res.warnings)


def test_warns_on_too_few_points():
    conc, signal = make_data(kd=10.0, n=5)
    res = fit(conc, signal)
    assert any("データ点が 5 点のみ" in w for w in res.warnings)


def test_warns_when_no_points_near_kd():
    """Nothing measured near Kd -> warning that the inflection point is not pinned down."""
    conc = np.array([0.0, 0.05, 0.1, 0.2, 0.5, 100.0, 300.0, 1000.0])
    signal = langmuir(conc, 10.0, 1.0, 0.0)
    res = fit(conc, signal)
    assert any("測定点がありません" in w for w in res.warnings)


def test_warns_when_no_points_below_kd():
    """An SPR-like design whose lowest concentration, 0.12 uM, is still above Kd = 0.047 uM.

    Every point sits on the saturated side, so Kd is set by extrapolating below the measured range.
    """
    conc = np.array([0.12, 0.25, 0.5, 1.0, 2.0])
    signal = langmuir(conc, 0.047, 1.0, 0.0)
    res = fit(conc, signal)
    assert conc.min() > res.location
    assert any("外挿で決まっている" in w for w in res.warnings)
    assert any("有効数字を増やして報告できません" in w for w in res.warnings)


def test_no_extrapolation_warning_when_low_points_exist():
    conc, signal = make_data(kd=10.0)
    res = fit(conc, signal)
    assert conc.min() == 0.0
    assert not any("外挿で決まっている" in w for w in res.warnings)


def test_warns_on_ligand_depletion():
    """Receptor concentration above one tenth of Kd -> warning that a tight-binding treatment is needed."""
    conc, signal = make_data(kd=10.0)
    res = fit(conc, signal, receptor_conc=5.0)
    assert any("tight-binding" in w for w in res.warnings)


def test_no_depletion_warning_when_receptor_is_dilute():
    conc, signal = make_data(kd=10.0)
    res = fit(conc, signal, receptor_conc=0.1)
    assert not any("tight-binding" in w for w in res.warnings)


def test_diagnose_uses_model_display_names():
    from bindfit import michaelis

    conc = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    msgs = diagnose(conc, np.zeros_like(conc), michaelis, {"km": 1.6, "vmax": 1.0, "baseline": 0.0})
    assert any("Km=" in m for m in msgs)
    assert not any("Kd=" in m for m in msgs)


# ------------------------------------------------------------------- CSV loading


def test_load_csv_skips_header_and_reads_values(tmp_path):
    path = tmp_path / "titration.csv"
    path.write_text(
        "concentration_nM,signal\n0,0.02\n1,0.11\n3,0.25\n10,0.51\n30,0.77\n100,0.93\n",
        encoding="utf-8",
    )
    conc, signal = load_csv(path)
    assert len(conc) == 6
    assert conc[0] == 0.0
    assert signal[-1] == pytest.approx(0.93)


def test_load_csv_rejects_too_few_rows(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("conc,signal\n1,0.1\n2,0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Only 2 data point"):
        load_csv(path)


def test_load_csv_rejects_negative_concentration(tmp_path):
    path = tmp_path / "neg.csv"
    path.write_text("conc,signal\n-1,0.1\n2,0.2\n5,0.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="negative values"):
        load_csv(path)


# ----------------------------------------------------------- plotting data and report


def test_report_contains_labels_and_unit():
    conc, signal = make_data(kd=10.0)
    text = fit(conc, signal, unit="nM").report()
    assert "Kd" in text and "nM" in text and "R^2" in text and "langmuir" in text


def test_predict_matches_model():
    conc, signal = make_data(kd=10.0)
    res = fit(conc, signal)
    assert res.predict(res.location) == pytest.approx(res.params["baseline"] + res.params["bmax"] / 2, rel=1e-9)
    np.testing.assert_allclose(res.predict(conc), langmuir(conc, *res.values))


def test_residuals_are_small_for_clean_data():
    conc, signal = make_data(kd=10.0)
    res = fit(conc, signal)
    assert np.abs(res.residuals(conc, signal)).max() < 1e-6


def test_curve_is_log_spaced_and_covers_kd():
    conc, signal = make_data(kd=10.0)
    res = fit(conc, signal)
    x, y = res.curve()
    assert len(x) == len(y) == 300
    assert x[0] == pytest.approx(res.location / 100)
    assert x[-1] == pytest.approx(res.location * 100)
    ratios = x[1:] / x[:-1]
    assert ratios.std() / ratios.mean() < 1e-9
    assert np.all(np.diff(y) > 0)


def test_curve_respects_explicit_range():
    conc, signal = make_data(kd=10.0)
    res = fit(conc, signal)
    x, _ = res.curve(0.5, 50.0, n=17)
    assert len(x) == 17
    assert x[0] == pytest.approx(0.5)
    assert x[-1] == pytest.approx(50.0)


def test_curve_rejects_nonpositive_and_inverted_range():
    conc, signal = make_data(kd=10.0)
    res = fit(conc, signal)
    with pytest.raises(ValueError, match="must be positive"):
        res.curve(0.0, 10.0)
    with pytest.raises(ValueError, match="greater than conc_min"):
        res.curve(10.0, 1.0)


def test_matplotlib_is_not_required_to_import_bindfit():
    """The library itself never touches matplotlib.

    Plotting is the job of the sample examples/plot_fit.py, which is not part of the distribution.
    """
    import subprocess
    import sys as _sys

    code = (
        "import sys\n"
        "sys.modules['matplotlib'] = None\n"
        "import bindfit\n"
        "import numpy as np\n"
        "c = np.array([0.0, 1, 3, 10, 30, 100, 300])\n"
        "s = bindfit.langmuir(c, 10.0, 1.0, 0.0)\n"
        "r = bindfit.fit(c, s)\n"
        "x, y = r.curve()\n"
        "assert abs(r.params['kd'] - 10.0) < 1e-6 and len(x) == 300\n"
        "assert 'matplotlib.pyplot' not in sys.modules\n"
        "print('ok')\n"
    )
    out = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
