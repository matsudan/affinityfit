"""Tests that no numbers are invented when a confidence interval cannot be computed.

Zero degrees of freedom and a rank-deficient Jacobian both break the premise that an
interval can be built from the covariance matrix. Left alone, either one surfaces as an
absurdly narrow interval.
"""

from __future__ import annotations

import numpy as np
import pytest

from affinityfit import Dataset, fit, fit_global, hill, langmuir
from affinityfit.fitting import _jacobian_rank

METHODS = ("asymptotic", "profile", "bootstrap")


# ------------------------------------------------------------ Zero degrees of freedom


@pytest.mark.parametrize("method", METHODS)
def test_no_degrees_of_freedom_yields_undetermined_intervals(method):
    """Three points for three parameters. By definition the curve passes through every point."""
    conc = np.array([1.0, 10.0, 100.0])
    signal = langmuir(conc, 10.0, 1.0, 0.0)
    res = fit(conc, signal, ci=method, n_boot=100)
    assert res.n_points == 3
    for name in langmuir.params:
        iv = res.intervals[name]
        assert not iv.bounded, (method, name, iv)
        assert "undetermined" in iv.format()


def test_no_degrees_of_freedom_is_explained_in_the_warnings():
    conc = np.array([1.0, 10.0, 100.0])
    res = fit(conc, langmuir(conc, 10.0, 1.0, 0.0), ci="asymptotic")
    message = next(w for w in res.warnings if "自由度" in w)
    assert "3 点" in message and "3 個" in message
    assert "信頼区間は算出できません" in message


def test_point_estimate_is_still_returned_when_there_is_no_spread():
    """No interval can be given, but the point estimate itself is correct."""
    conc = np.array([1.0, 10.0, 100.0])
    res = fit(conc, langmuir(conc, 10.0, 1.0, 0.0), ci="asymptotic")
    assert res.params["kd"] == pytest.approx(10.0, rel=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_four_parameters_on_four_points_is_also_caught(method):
    conc = np.array([1.0, 5.0, 20.0, 100.0])
    res = fit(conc, hill(conc, 10.0, 1.0, 0.0, 2.0), model=hill, ci=method, n_boot=100)
    assert not res.intervals["n"].bounded
    assert any("自由度" in w for w in res.warnings)


def test_fixing_a_parameter_restores_the_degrees_of_freedom():
    """Fixing a parameter creates a degree of freedom, and the intervals appear."""
    conc = np.array([1.0, 10.0, 100.0])
    signal = langmuir(conc, 10.0, 1.0, 0.0)
    res = fit(conc, signal, fixed={"baseline": 0.0}, ci="asymptotic")
    assert res.intervals["kd"].bounded
    assert not any("自由度" in w for w in res.warnings)


def test_one_more_point_restores_the_degrees_of_freedom():
    conc = np.array([1.0, 3.0, 10.0, 100.0])
    res = fit(conc, langmuir(conc, 10.0, 1.0, 0.0), ci="asymptotic")
    assert res.intervals["kd"].bounded
    assert not any("自由度" in w for w in res.warnings)


def test_global_fit_counts_shared_parameters_when_judging_the_degrees_of_freedom():
    """Sharing reduces the number of estimated parameters, so the same points now leave degrees of freedom."""
    conc = np.array([1.0, 10.0, 100.0])
    a = langmuir(conc, 10.0, 1.0, 0.0)
    b = langmuir(conc, 40.0, 1.0, 0.0)
    datasets = [Dataset("a", conc, a), Dataset("b", conc, b)]

    free = fit_global(datasets, fixed={"baseline": 0.0}, ci="asymptotic")
    assert free.n_free_params == 4  # kd x2 + bmax x2, so 2 degrees of freedom over 6 points
    assert free.intervals["a"]["kd"].bounded

    shared = fit_global(datasets, shared=["bmax"], fixed={"baseline": 0.0}, ci="asymptotic")
    assert shared.n_free_params == 3
    assert shared.intervals["a"]["kd"].bounded


# --------------------------------------------------- Detecting rank deficiency


def test_rank_deficient_jacobian_is_detected_and_not_papered_over():
    """With only two distinct concentrations, three parameters cannot be told apart.

    There are 3 degrees of freedom here, so this is a path the degrees-of-freedom check does not catch.
    """
    conc = np.array([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
    signal = langmuir(conc, 10.0, 1.0, 0.0)
    res = fit(conc, signal, ci="asymptotic")
    assert res.n_points - 3 > 0  # there are degrees of freedom
    assert not res.intervals["kd"].bounded
    message = next(w for w in res.warnings if "ランク" in w)
    assert "一意に" in message


def test_rank_warning_names_the_rank_and_the_parameter_count():
    conc = np.array([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
    res = fit(conc, langmuir(conc, 10.0, 1.0, 0.0), ci="asymptotic")
    message = next(w for w in res.warnings if "ランク" in w)
    assert "ランクが 2" in message and "3 個" in message


def test_three_distinct_concentrations_are_enough_for_three_parameters():
    """Guard against false alarms: no warning when the distinct concentrations cover the parameter count."""
    conc = np.array([1.0, 1.0, 10.0, 10.0, 100.0, 100.0])
    res = fit(conc, langmuir(conc, 10.0, 1.0, 0.0), ci="asymptotic")
    assert not any("ランク" in w for w in res.warnings)


def test_healthy_titration_has_neither_warning():
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 12)])
    signal = langmuir(conc, 10.0, 1.0, 0.02) + np.random.default_rng(0).normal(0, 0.01, conc.size)
    res = fit(conc, signal, ci="asymptotic")
    assert res.warnings == ()
    assert res.intervals["kd"].bounded


# --------------------------------------------- Properties of the rank computation itself


def test_jacobian_rank_normalises_columns_before_judging():
    """Columns whose magnitudes differ by orders of magnitude must still count as full rank when independent."""
    jac = np.array([[1e-12, 1.0], [2e-12, 3.0], [3e-12, 7.0]])
    assert _jacobian_rank(jac) == 2


def test_jacobian_rank_detects_a_duplicated_direction():
    jac = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 3.0], [3.0, 6.0, 7.0]])  # column 2 = column 1 x2
    assert _jacobian_rank(jac) == 2


def test_jacobian_rank_treats_a_dead_column_as_deficient():
    jac = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert _jacobian_rank(jac) == 1
