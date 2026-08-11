"""Structured diagnostics retain the conditions that make their advice valid."""

from __future__ import annotations

import numpy as np
import pytest

from affinityfit import Dataset, fit, fit_global, hill, langmuir, michaelis
from affinityfit.uncertainty import MIN_BOOTSTRAP_SAMPLES

CONC = np.concatenate([[0.0], np.logspace(-1, 3, 12)])
SIGNAL = langmuir(CONC, 10.0, 1.0, 0.02) + np.random.default_rng(0).normal(0, 0.01, CONC.size)
NARROW = np.array([0.2, 0.32, 0.5, 0.8, 1.1, 1.6])


def codes(diagnostics):
    return {diagnostic.code for diagnostic in diagnostics}


def test_reports_keep_aicc_visible_for_human_use():
    single = fit(CONC, SIGNAL, ci="asymptotic")
    combined = fit_global([Dataset("d", CONC, SIGNAL)], ci="asymptotic")
    assert single.aicc == pytest.approx(combined.aicc)
    assert "AICc" in single.report()
    assert "AICc" in combined.report()


def test_hill_advice_has_a_stable_code():
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 15)])
    signal = langmuir(conc, 10.0, 1.0, 0.0) + np.random.default_rng(3).normal(0, 0.02, conc.size)
    assert "hill_n_includes_one" in codes(fit(conc, signal, model=hill, ci="asymptotic").warnings)


def broken_and_healthy(noise_b):
    rng = np.random.default_rng(0)
    a = langmuir(CONC, 5.0, 1.0, 0.0) + rng.normal(0, 0.02, CONC.size)
    b = langmuir(NARROW, 9.0, 1.0, 0.0) + np.random.default_rng(1).normal(0, noise_b, NARROW.size)
    return [Dataset("a", CONC, a), Dataset("b", NARROW, b)]


def test_sharing_note_is_suppressed_when_the_fit_is_broken():
    res = fit_global(broken_and_healthy(noise_b=0.30), shared=["bmax"], fixed={"baseline": 0.0}, ci="asymptotic")
    assert "no_fit" in codes(res.diagnostics_per["b"])
    assert "shared_amplitude_identifies_location" not in codes(res.diagnostics_per["b"])


def test_sharing_note_remains_when_the_fit_is_sound():
    res = fit_global(broken_and_healthy(noise_b=0.01), shared=["bmax"], fixed={"baseline": 0.0}, ci="asymptotic")
    assert "no_fit" not in codes(res.diagnostics_per["b"])
    assert "shared_amplitude_identifies_location" in codes(res.diagnostics_per["b"])


def test_amplitude_collapse_also_suppresses_the_sharing_note():
    down1 = langmuir(CONC, 10.0, -0.8, 1.0)
    down2 = langmuir(NARROW, 9.0, -0.8, 1.0)
    res = fit_global(
        [Dataset("down1", CONC, down1), Dataset("down2", NARROW, down2)],
        model=michaelis,
        shared=["vmax"],
        fixed={"baseline": 1.0},
        ci="asymptotic",
    )
    for name in res.names:
        assert "amplitude_collapsed" in codes(res.diagnostics_per[name])
        assert "shared_amplitude_identifies_location" not in codes(res.diagnostics_per[name])


@pytest.mark.parametrize("n_boot", [1, 10, 50, MIN_BOOTSTRAP_SAMPLES - 1])
def test_too_few_resamples_are_rejected(n_boot):
    with pytest.raises(ValueError, match="below the 100 resamples"):
        fit(CONC, SIGNAL, ci="bootstrap", n_boot=n_boot)


def test_the_minimum_bootstrap_count_is_accepted():
    assert fit(CONC, SIGNAL, ci="bootstrap", n_boot=MIN_BOOTSTRAP_SAMPLES).intervals["kd"].bounded
