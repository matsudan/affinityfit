"""Rank and degree-of-freedom failures are exposed as structured diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from affinityfit import Dataset, fit, fit_global, hill, langmuir
from affinityfit.intervals import _jacobian_rank

METHODS = ("asymptotic", "profile", "bootstrap")


def codes(result):
    return {diagnostic.code for diagnostic in result.diagnostics}


@pytest.mark.parametrize("method", METHODS)
def test_no_degrees_of_freedom_yields_undetermined_intervals(method):
    conc = np.array([1.0, 10.0, 100.0])
    res = fit(conc, langmuir(conc, 10.0, 1.0, 0.0), ci=method, n_boot=100)
    assert all(not res.intervals[name].bounded for name in langmuir.params)
    assert "no_degrees_of_freedom" in codes(res)


def test_point_estimate_is_returned_without_degrees_of_freedom():
    conc = np.array([1.0, 10.0, 100.0])
    assert fit(conc, langmuir(conc, 10.0, 1.0, 0.0), ci="asymptotic").params["kd"] == pytest.approx(10.0, rel=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_four_parameters_on_four_points_is_also_caught(method):
    conc = np.array([1.0, 5.0, 20.0, 100.0])
    res = fit(conc, hill(conc, 10.0, 1.0, 0.0, 2.0), model=hill, ci=method, n_boot=100)
    assert not res.intervals["n"].bounded
    assert "no_degrees_of_freedom" in codes(res)


def test_fixing_a_parameter_restores_degrees_of_freedom():
    conc = np.array([1.0, 10.0, 100.0])
    res = fit(conc, langmuir(conc, 10.0, 1.0, 0.0), fixed={"baseline": 0.0}, ci="asymptotic")
    assert res.intervals["kd"].bounded
    assert "no_degrees_of_freedom" not in codes(res)


def test_global_fit_counts_shared_parameters_when_judging_degrees_of_freedom():
    conc = np.array([1.0, 10.0, 100.0])
    datasets = [Dataset("a", conc, langmuir(conc, 10.0, 1.0, 0.0)), Dataset("b", conc, langmuir(conc, 40.0, 1.0, 0.0))]
    res = fit_global(datasets, shared=["bmax"], fixed={"baseline": 0.0}, ci="asymptotic")
    assert res.n_free_params == 3
    assert res.intervals["a"]["kd"].bounded


def test_rank_deficient_jacobian_is_reported():
    conc = np.array([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
    res = fit(conc, langmuir(conc, 10.0, 1.0, 0.0), ci="asymptotic")
    assert not res.intervals["kd"].bounded
    assert "rank_deficient_jacobian" in codes(res)


def test_three_distinct_concentrations_are_enough_for_three_parameters():
    conc = np.array([1.0, 1.0, 10.0, 10.0, 100.0, 100.0])
    assert "rank_deficient_jacobian" not in codes(fit(conc, langmuir(conc, 10.0, 1.0, 0.0), ci="asymptotic"))


def test_jacobian_rank_normalises_columns_before_judging():
    jac = np.array([[1e-12, 1.0], [2e-12, 3.0], [3e-12, 7.0]])
    assert _jacobian_rank(jac) == 2


def test_jacobian_rank_detects_a_duplicated_direction():
    jac = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 3.0], [3.0, 6.0, 7.0]])
    assert _jacobian_rank(jac) == 2


def test_jacobian_rank_treats_a_dead_column_as_deficient():
    assert _jacobian_rank(np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])) == 1
