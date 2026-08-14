"""Observables that decrease on binding, such as fluorescence quenching or a falling intensity.

A saturation curve still fits as a flat line once the amplitude has collapsed to 0. Restricting the amplitude to
non-negative values on decreasing data therefore returns a plausible-looking Kd while nothing has been fitted at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from affinityfit import Dataset, fit, fit_global, hill, langmuir, michaelis
from affinityfit.models import _is_decreasing

CONC = np.concatenate([[0.0], np.logspace(-1, 3, 11)])
# fluorescence quenching: starts from a baseline of 1.0 and falls to 0.2 on binding
KD, BMAX, BASELINE = 10.0, -0.8, 1.0


def quench(noise=0.0, seed=0, kd=KD, bmax=BMAX, baseline=BASELINE):
    signal = langmuir(CONC, kd, bmax, baseline)
    if noise:
        signal = signal + np.random.default_rng(seed).normal(0, noise * abs(bmax), CONC.size)
    return CONC, signal


# --------------------------------------------------------- judging the trend


def test_is_decreasing_detects_direction():
    conc, signal = quench()
    assert _is_decreasing(conc, signal)
    assert not _is_decreasing(conc, langmuir(conc, KD, +0.8, 0.0))


def test_is_decreasing_is_robust_to_noise_on_single_points():
    conc, signal = quench(noise=0.15, seed=2)
    assert _is_decreasing(conc, signal)


def test_initial_guess_picks_a_negative_amplitude_for_decreasing_data():
    conc, signal = quench()
    assert langmuir.initial(conc, signal)["bmax"] < 0
    assert langmuir.initial(conc, langmuir(conc, KD, +0.8, 0.0))["bmax"] > 0


# ------------------------------------------------- recovery on decreasing data


def test_langmuir_recovers_decreasing_parameters_exactly():
    conc, signal = quench()
    res = fit(conc, signal)
    assert res.params["kd"] == pytest.approx(KD, rel=1e-6)
    assert res.params["bmax"] == pytest.approx(BMAX, rel=1e-6)
    assert res.params["baseline"] == pytest.approx(BASELINE, rel=1e-6)
    assert res.r_squared > 0.9999


def test_hill_recovers_decreasing_parameters():
    conc = CONC
    signal = hill(conc, KD, BMAX, BASELINE, 2.0)
    res = fit(conc, signal, model=hill)
    assert res.params["kd"] == pytest.approx(KD, rel=1e-3)
    assert res.params["bmax"] == pytest.approx(BMAX, rel=1e-3)
    assert res.params["n"] == pytest.approx(2.0, rel=1e-3)


def test_recovers_decreasing_parameters_with_noise():
    conc, signal = quench(noise=0.03, seed=5)
    res = fit(conc, signal)
    assert res.params["kd"] == pytest.approx(KD, rel=0.3)
    assert res.params["bmax"] < 0
    assert res.intervals["bmax"].contains(BMAX)


def test_decreasing_data_produces_no_spurious_model_warnings():
    conc, signal = quench(noise=0.02, seed=1)
    codes = {diagnostic.code for diagnostic in fit(conc, signal).warnings}
    assert "amplitude_collapsed" not in codes
    assert "no_fit" not in codes


def test_profile_interval_handles_a_negative_amplitude():
    conc, signal = quench(noise=0.02, seed=3)
    iv = fit(conc, signal, ci="profile").intervals["bmax"]
    assert iv.bounded
    assert iv.upper is not None
    assert iv.upper < 0
    assert iv.contains(BMAX)


def test_global_fit_shares_a_negative_amplitude():
    signal_a = langmuir(CONC, 5.0, BMAX, BASELINE)
    signal_b = langmuir(CONC, 50.0, BMAX, BASELINE)
    res = fit_global(
        [Dataset("a", CONC, signal_a), Dataset("b", CONC, signal_b)],
        shared=["bmax"],
        ci="asymptotic",
    )
    assert res.params["a"]["bmax"] == res.params["b"]["bmax"]
    assert res.params["a"]["bmax"] == pytest.approx(BMAX, rel=1e-4)
    assert res.params["a"]["kd"] == pytest.approx(5.0, rel=1e-3)
    assert res.params["b"]["kd"] == pytest.approx(50.0, rel=1e-3)


# ------------------------------------ not staying silent when the model cannot express the data


def test_michaelis_on_decreasing_data_says_so_and_points_at_the_fix():
    """The structured diagnostic names the collapsed-amplitude condition without string matching."""
    conc, signal = quench()
    codes = {diagnostic.code for diagnostic in fit(conc, signal, model=michaelis).warnings}
    assert "amplitude_collapsed" in codes
    assert "no_fit" in codes


def test_no_fit_is_reported_for_data_the_model_cannot_describe():
    conc = np.linspace(1.0, 20.0, 12)
    signal = np.array([0.5, 2.0, 0.1, 1.8, 0.3, 2.2, 0.4, 1.9, 0.2, 2.1, 0.6, 1.7])
    res = fit(conc, signal)
    assert res.r_squared < 0.5
    assert "no_fit" in {diagnostic.code for diagnostic in res.warnings}


def test_amplitude_collapse_is_machine_readable():
    conc, signal = quench()
    res = fit(conc, signal, model=michaelis)
    assert "amplitude_collapsed" in {diagnostic.code for diagnostic in res.warnings}


def test_no_fit_warning_absent_for_an_exact_fit():
    from affinityfit import diagnose

    conc, signal = quench()
    diagnostics = diagnose(conc, signal, langmuir, {"kd": KD, "bmax": BMAX, "baseline": BASELINE})
    assert "no_fit" not in {diagnostic.code for diagnostic in diagnostics}


def test_good_fit_does_not_trigger_the_no_fit_warning():
    conc, signal = quench(noise=0.05, seed=7)
    res = fit(conc, signal)
    assert res.r_squared > 0.9
    assert "no_fit" not in {diagnostic.code for diagnostic in res.warnings}
