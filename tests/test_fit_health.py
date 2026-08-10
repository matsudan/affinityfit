"""Health checks on the fit itself.

The diagnostics about where the measurements sit (saturation never reached, no points
near Kd, and so on) stay silent when the model does not match the mechanism. A high
coefficient of determination does not rescue a fit whose residual signs are skewed,
and a parameter stuck at a bound is not an estimate.
"""

from __future__ import annotations

import numpy as np
import pytest

from bindfit import diagnose, fit, hill, langmuir, michaelis
from bindfit.core import _residual_structure

CONC = np.concatenate([[0.0], np.logspace(-1, 3, 15)])


def cooperative(n=3.0, noise=0.02, seed=0):
    """Genuinely cooperative data. Fitting it with a 1:1 model deviates systematically."""
    return CONC, hill(CONC, 10.0, 1.0, 0.0, n) + np.random.default_rng(seed).normal(0, noise, CONC.size)


# ------------------------------------------------- Systematic structure in the residuals


def test_wrong_model_with_high_r_squared_is_caught():
    """Cooperative data fitted with langmuir gives R^2 = 0.96: the coefficient of determination lets it pass."""
    conc, signal = cooperative(n=3.0)
    res = fit(conc, signal, model=langmuir)
    assert res.r_squared > 0.9  # on the coefficient of determination alone it looks acceptable
    assert any("残差が系統的に偏っています" in w for w in res.warnings)


def test_correct_model_on_the_same_data_is_silent():
    """Guard against false alarms: the correct model raises no residual warning."""
    conc, signal = cooperative(n=3.0)
    res = fit(conc, signal, model=hill)
    assert not any("残差" in w for w in res.warnings)


@pytest.mark.parametrize("n_true", [2.0, 3.0, 5.0])
def test_systematic_deviation_is_caught_across_cooperativity(n_true):
    conc, signal = cooperative(n=n_true, seed=2)
    res = fit(conc, signal, model=langmuir)
    assert any("残差" in w for w in res.warnings), res.r_squared


def test_all_residuals_of_one_sign_is_reported():
    params = {"kd": 10.0, "bmax": 1.0, "baseline": 0.0}
    signal = langmuir(CONC, *langmuir.ordered(params)) + 0.5  # shift everything in one direction
    msgs = _residual_structure(CONC, signal, langmuir, params)
    assert msgs and "すべてが同じ符号" in msgs[0][1]


def test_residual_test_is_skipped_for_an_essentially_exact_fit():
    """When the residuals are at the level of floating-point rounding, their sign pattern means nothing."""
    params = {"kd": 10.0, "bmax": 1.0, "baseline": 0.0}
    signal = langmuir(CONC, *langmuir.ordered(params))
    assert _residual_structure(CONC, signal, langmuir, params) == []


def test_residual_test_is_skipped_for_too_few_points():
    conc = np.array([0.0, 1.0, 3.0, 10.0, 30.0])
    params = {"kd": 10.0, "bmax": 1.0, "baseline": 0.0}
    signal = langmuir(conc, *langmuir.ordered(params)) + 0.3
    assert _residual_structure(conc, signal, langmuir, params) == []


def test_residual_message_reports_the_runs_statistic():
    conc, signal = cooperative(n=3.0)
    res = fit(conc, signal, model=langmuir)
    message = next(w for w in res.warnings if "残差" in w)
    assert "符号の連" in message and "z = " in message and "自己相関" in message
    assert "+" in message and "-" in message  # the sign pattern is shown


def test_residuals_are_ordered_by_concentration_not_by_input_order():
    """The verdict must not depend on input order: shuffling the points gives the same result."""
    conc, signal = cooperative(n=3.0)
    shuffle = np.random.default_rng(7).permutation(conc.size)
    a = fit(conc, signal, model=langmuir).warnings
    b = fit(conc[shuffle], signal[shuffle], model=langmuir).warnings
    assert any("残差" in w for w in a)
    assert any("残差" in w for w in b)


# ------------------------------------------- Parameters stuck at a bound


def test_parameter_stuck_at_its_upper_bound_is_reported():
    """Step-like data drives the Hill coefficient onto its upper bound of 20."""
    step = np.where(CONC < 10.0, 0.0, 1.0) + np.random.default_rng(1).normal(0, 0.01, CONC.size)
    res = fit(CONC, step, model=hill)
    assert res.params["n"] == pytest.approx(hill.bounds["n"][1])
    stuck = [w for w in res.warnings if "張り付いています" in w]
    assert stuck
    assert "上限" in stuck[0] and "制約の産物" in stuck[0]


def test_parameter_stuck_at_its_lower_bound_is_reported():
    """Feeding decreasing data to michaelis pins Vmax at its lower bound of 0."""
    signal = langmuir(CONC, 10.0, -0.8, 1.0)
    res = fit(CONC, signal, model=michaelis)
    assert any("潰れています" in w for w in res.warnings)


def test_fixed_parameters_are_exempt_from_the_bound_check():
    """A fixed value that coincides with a bound is what was asked for, so it draws no warning."""
    signal = langmuir(CONC, 10.0, 1.0, 0.0) + np.random.default_rng(3).normal(0, 0.01, CONC.size)
    res = fit(CONC, signal, fixed={"baseline": 0.0})
    assert not any("張り付いています" in w for w in res.warnings)


def test_healthy_fit_has_no_bound_warning():
    signal = langmuir(CONC, 10.0, 1.0, 0.02) + np.random.default_rng(4).normal(0, 0.01, CONC.size)
    res = fit(CONC, signal, model=hill)
    assert not any("張り付いています" in w for w in res.warnings)


# --------------------------------------------------- Two tiers for the coefficient of determination


def test_low_r_squared_is_reported_as_advisory():
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 11)])
    signal = langmuir(conc, 10.0, 1.0, 0.0) + np.random.default_rng(5).normal(0, 0.18, conc.size)
    res = fit(conc, signal)
    assert 0.5 < res.r_squared < 0.9
    assert any("飽和曲線としては低め" in w for w in res.warnings)


def test_r_squared_near_zero_says_the_value_is_meaningless():
    conc = np.linspace(1.0, 20.0, 12)
    signal = np.array([0.5, 2.0, 0.1, 1.8, 0.3, 2.2, 0.4, 1.9, 0.2, 2.1, 0.6, 1.7])
    res = fit(conc, signal)
    assert res.r_squared < 0.5
    assert any("意味はありません" in w for w in res.warnings)


def test_high_r_squared_produces_no_r_squared_warning():
    signal = langmuir(CONC, 10.0, 1.0, 0.02) + np.random.default_rng(6).normal(0, 0.01, CONC.size)
    res = fit(CONC, signal)
    assert not any("決定係数" in w for w in res.warnings)


# ------------------------------------------- The signal argument actually matters


def test_signal_argument_changes_the_diagnosis():
    """The signal passed to diagnose affects the result; it is not a decorative argument.

    For the same parameters, a matching signal and an offset signal must lead to different verdicts.
    """
    params = {"kd": 10.0, "bmax": 1.0, "baseline": 0.0}
    exact = langmuir(CONC, *langmuir.ordered(params))
    offset = exact + 0.5  # observations offset from the curve in one direction

    good = diagnose(CONC, exact, langmuir, params, r_squared=1.0)
    bad = diagnose(CONC, offset, langmuir, params, r_squared=0.5)
    assert not any("残差" in m for m in good)
    assert any("残差" in m for m in bad)


def test_zero_signal_is_flagged_rather_than_silently_accepted():
    """Back when signal was ignored, even a zero vector passed."""
    params = {"kd": 10.0, "bmax": 1.0, "baseline": 0.0}
    msgs = diagnose(CONC, np.zeros_like(CONC), langmuir, params, r_squared=0.0)
    assert any("残差" in m for m in msgs)
    assert any("意味はありません" in m for m in msgs)
