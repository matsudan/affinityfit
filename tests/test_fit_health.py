"""Fit-health diagnostics expose stable codes instead of localized prose."""

from __future__ import annotations

import numpy as np
import pytest

from affinityfit import diagnose, fit, hill, langmuir, michaelis
from affinityfit.core import _residual_structure

CONC = np.concatenate([[0.0], np.logspace(-1, 3, 15)])


def cooperative(n=3.0, noise=0.02, seed=0):
    return CONC, hill(CONC, 10.0, 1.0, 0.0, n) + np.random.default_rng(seed).normal(0, noise, CONC.size)


def codes(result):
    return {diagnostic.code for diagnostic in result.diagnostics}


# ------------------------------------------------- systematic residual structure


def test_wrong_model_with_high_r_squared_is_caught():
    conc, signal = cooperative(n=3.0)
    res = fit(conc, signal, model=langmuir)
    assert res.r_squared > 0.9
    diagnostic = next(diagnostic for diagnostic in res.diagnostics if diagnostic.code == "residual_structure")
    assert diagnostic.severity == "warning"
    assert diagnostic.message.isascii()


def test_correct_model_on_the_same_data_is_silent():
    conc, signal = cooperative(n=3.0)
    assert "residual_structure" not in codes(fit(conc, signal, model=hill))


@pytest.mark.parametrize("n_true", [2.0, 3.0, 5.0])
def test_systematic_deviation_is_caught_across_cooperativity(n_true):
    conc, signal = cooperative(n=n_true, seed=2)
    assert "residual_structure" in codes(fit(conc, signal, model=langmuir))


def test_residual_test_is_skipped_for_an_essentially_exact_fit():
    params = {"kd": 10.0, "bmax": 1.0, "baseline": 0.0}
    assert _residual_structure(CONC, langmuir(CONC, *langmuir.ordered(params)), langmuir, params) == []


def test_residual_test_is_skipped_for_too_few_points():
    conc = np.array([0.0, 1.0, 3.0, 10.0, 30.0])
    params = {"kd": 10.0, "bmax": 1.0, "baseline": 0.0}
    assert _residual_structure(conc, langmuir(conc, *langmuir.ordered(params)) + 0.3, langmuir, params) == []


def test_residual_verdict_is_order_invariant():
    conc, signal = cooperative(n=3.0)
    shuffle = np.random.default_rng(7).permutation(conc.size)
    assert "residual_structure" in codes(fit(conc, signal, model=langmuir))
    assert "residual_structure" in codes(fit(conc[shuffle], signal[shuffle], model=langmuir))


# ---------------------------------------------------- parameters at bounds


def test_parameter_stuck_at_its_upper_bound_is_reported():
    step = np.where(CONC < 10.0, 0.0, 1.0) + np.random.default_rng(1).normal(0, 0.01, CONC.size)
    res = fit(CONC, step, model=hill)
    assert res.params["n"] == pytest.approx(hill.bounds["n"][1])
    assert "param_at_bound" in codes(res)


def test_parameter_stuck_at_its_lower_bound_is_reported():
    signal = langmuir(CONC, 10.0, -0.8, 1.0)
    assert "amplitude_collapsed" in codes(fit(CONC, signal, model=michaelis))


def test_fixed_parameters_are_exempt_from_the_bound_check():
    signal = langmuir(CONC, 10.0, 1.0, 0.0) + np.random.default_rng(3).normal(0, 0.01, CONC.size)
    assert "param_at_bound" not in codes(fit(CONC, signal, fixed={"baseline": 0.0}))


def test_healthy_fit_has_no_bound_warning():
    signal = langmuir(CONC, 10.0, 1.0, 0.02) + np.random.default_rng(4).normal(0, 0.01, CONC.size)
    assert "param_at_bound" not in codes(fit(CONC, signal, model=hill))


# --------------------------------------------------------------- fit quality


def test_a_model_indistinguishable_from_its_own_mean_is_flagged():
    """Data with no trend at all: the fitted model is no better than a flat line at the mean."""
    conc = np.linspace(1.0, 20.0, 12)
    signal = np.array([0.5, 2.0, 0.1, 1.8, 0.3, 2.2, 0.4, 1.9, 0.2, 2.1, 0.6, 1.7])
    res = fit(conc, signal)
    assert "no_fit" in codes(res)


def test_a_noisy_but_genuine_fit_is_not_flagged_as_no_fit():
    """A high enough noise level lowers R^2 without making the model indistinguishable from its mean."""
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 11)])
    signal = langmuir(conc, 10.0, 1.0, 0.0) + np.random.default_rng(5).normal(0, 0.18, conc.size)
    res = fit(conc, signal)
    assert res.r_squared < 0.9
    assert "no_fit" not in codes(res)


def test_high_r_squared_produces_no_fit_warning():
    signal = langmuir(CONC, 10.0, 1.0, 0.02) + np.random.default_rng(6).normal(0, 0.01, CONC.size)
    assert "no_fit" not in codes(fit(CONC, signal))


# --------------------------------------------------------------- diagnose()


def test_signal_argument_changes_the_diagnosis():
    params = {"kd": 10.0, "bmax": 1.0, "baseline": 0.0}
    exact = langmuir(CONC, *langmuir.ordered(params))
    good = diagnose(CONC, exact, langmuir, params)
    bad = diagnose(CONC, exact + 0.5, langmuir, params)
    assert "residual_structure" not in {diagnostic.code for diagnostic in good}
    assert "residual_structure" in {diagnostic.code for diagnostic in bad}


def test_zero_signal_is_flagged_rather_than_silently_accepted():
    params = {"kd": 10.0, "bmax": 1.0, "baseline": 0.0}
    diagnostics = diagnose(CONC, np.zeros_like(CONC), langmuir, params)
    assert {"residual_structure", "no_fit"} <= {diagnostic.code for diagnostic in diagnostics}
