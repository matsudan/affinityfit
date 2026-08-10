"""Invariance with respect to the concentration scale.

Parameterising Kd linearly makes the fitted result depend on the choice of unit: the same
affinity written as 10 pM and as 1e-11 M gives different answers. Handling a picomolar affinity
in molar units is routine in SPR, so this is a mine that gets stepped on in practice.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from bindfit import Dataset, fit, fit_global, fitting, hill, langmuir, michaelis
from bindfit.fitting import _Problem
from bindfit.models import Model

# From fM to 100 M in molar units. Includes the picomolar affinity of SPR (1e-12 M).
DECADES = [10.0**e for e in range(2, -16, -1)]


def titration(kd, bmax=1.0, baseline=0.0, points=12, model=langmuir, extra=()):
    conc = np.concatenate([[0.0], np.logspace(np.log10(kd / 30), np.log10(kd * 30), points - 1)])
    return conc, model(conc, kd, bmax, baseline, *extra)


# ------------------------------------------------- Scale-invariant recovery


@pytest.mark.parametrize("kd", DECADES)
def test_recovers_kd_across_18_orders_of_magnitude(kd):
    conc, signal = titration(kd)
    res = fit(conc, signal, ci="asymptotic")
    assert res.params["kd"] == pytest.approx(kd, rel=1e-6)
    assert res.r_squared > 0.9999


@pytest.mark.parametrize("kd", [1e-12, 1e-11, 1e-10, 1e-9])
def test_picomolar_affinity_in_molar_units_with_noise(kd):
    """The real situation of SPR: a picomolar affinity handled in molar units."""
    conc, clean = titration(kd, points=14)
    signal = clean + np.random.default_rng(0).normal(0, 0.02, conc.size)
    res = fit(conc, signal, ci="asymptotic")
    assert res.params["kd"] == pytest.approx(kd, rel=0.2)
    assert res.intervals["kd"].contains(kd)


def test_unit_change_does_not_change_the_answer():
    """Writing the same experiment in M, nM or pM leaves the relative value of Kd unchanged."""
    kd_molar = 3.7e-11
    conc, signal = titration(kd_molar)
    results = {}
    for name, factor in (("M", 1.0), ("nM", 1e9), ("pM", 1e12)):
        res = fit(conc * factor, signal, ci="asymptotic")
        results[name] = res.params["kd"] / factor
    assert results["M"] == pytest.approx(results["nM"], rel=1e-6)
    assert results["M"] == pytest.approx(results["pM"], rel=1e-6)
    assert results["M"] == pytest.approx(kd_molar, rel=1e-6)


@pytest.mark.parametrize("km", [1e-12, 1e-6, 1.0])
def test_michaelis_is_also_scale_invariant(km):
    conc, signal = titration(km, bmax=0.014, model=michaelis)
    res = fit(conc, signal, model=michaelis, ci="asymptotic")
    assert res.params["km"] == pytest.approx(km, rel=1e-6)


@pytest.mark.parametrize("kd", [1e-12, 1e-6, 1.0])
def test_hill_is_also_scale_invariant(kd):
    conc, signal = titration(kd, model=hill, extra=(2.0,))
    res = fit(conc, signal, model=hill, ci="asymptotic")
    assert res.params["kd"] == pytest.approx(kd, rel=1e-4)
    assert res.params["n"] == pytest.approx(2.0, rel=1e-3)


def test_global_fit_is_scale_invariant():
    conc, sig_a = titration(1e-12)
    conc_b = conc
    sig_b = langmuir(conc_b, 1e-11, 1.0, 0.0)
    res = fit_global(
        [Dataset("tight", conc, sig_a), Dataset("weak", conc_b, sig_b)],
        shared=["bmax"],
        fixed={"baseline": 0.0},
        ci="asymptotic",
    )
    assert res.params["tight"]["kd"] == pytest.approx(1e-12, rel=1e-3)
    assert res.params["weak"]["kd"] == pytest.approx(1e-11, rel=1e-3)


# ------------------------------- Declaring the log scale, and handling the bounds


def test_only_positive_unbounded_parameters_use_the_log_scale():
    assert langmuir.is_log_scale("kd")
    assert michaelis.is_log_scale("km")
    assert not langmuir.is_log_scale("bmax")  # can be negative
    assert not langmuir.is_log_scale("baseline")
    assert not hill.is_log_scale("n")  # the upper bound is finite
    assert not michaelis.is_log_scale("vmax")  # the lower bound is 0


def test_profile_walk_uses_the_models_definition_not_the_bounds(monkeypatch):
    """The profile likelihood walk consults `Model.is_log_scale`.

    Rebuilding the decision from the bounds makes the treatment of the Hill coefficient, whose upper
    bound is finite, disagree with what the model declares. The values actually passed are captured to
    pin the agreement down.
    """
    recorded: list[bool] = []
    original = fitting.profile_bounds

    def spy(*args, **kwargs):
        recorded.append(bool(kwargs["log_scale"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(fitting, "profile_bounds", spy)

    conc = np.concatenate([[0.0], np.logspace(-1, 3, 15)])
    signal = hill(conc, 10.0, 1.0, 0.0, 2.0) + np.random.default_rng(0).normal(0, 0.01, conc.size)
    problem = _Problem([Dataset("d", conc, signal)], hill, (), {})
    fit(conc, signal, model=hill, ci="profile")

    assert recorded == [hill.is_log_scale(name) for name in problem.slot_param]
    assert recorded == [True, False, False, False]  # kd alone is logarithmic


def test_profile_endpoints_hold_for_a_finite_bounded_parameter():
    """The Hill coefficient, walked linearly, still meets the definition of the interval (the F-test threshold)."""
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 15)])
    signal = hill(conc, 10.0, 1.0, 0.0, 2.0) + np.random.default_rng(1).normal(0, 0.01, conc.size)
    res = fit(conc, signal, model=hill, ci="profile")

    def ssr(fixed):
        result = fit_global([Dataset("d", conc, signal)], model=hill, fixed=fixed)
        resid = result.result_for("d").residuals(conc, signal)
        return float(resid @ resid)

    ssr_min = ssr({})
    dof = len(conc) - 4
    threshold = ssr_min * (1.0 + float(stats.f.ppf(0.95, 1, dof)) / dof)
    interval = res.intervals["n"]
    assert interval.lower is not None and interval.upper is not None
    for edge in (interval.lower, interval.upper):
        assert ssr({"n": edge}) == pytest.approx(threshold, rel=1e-4)


def test_kd_lower_bound_does_not_truncate_picomolar_in_molar_units():
    """With a lower bound of 1e-12, a Kd of 1e-12 M landed exactly on that bound."""
    assert langmuir.lower("kd") < 1e-15


def test_no_spurious_bound_warning_for_small_kd():
    """Back when the bound check used an absolute tolerance, every Kd <= 1e-6 came out stuck at the lower bound."""
    for kd in (1e-4, 1e-6, 1e-12, 1e-16):
        conc, signal = titration(kd)
        res = fit(conc, signal, ci="asymptotic")
        assert not any("張り付いています" in w for w in res.warnings), kd


def test_bound_warning_still_fires_on_a_genuinely_pinned_parameter():
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 15)])
    step = np.where(conc < 10.0, 0.0, 1.0) + np.random.default_rng(1).normal(0, 0.01, conc.size)
    res = fit(conc, step, model=hill)
    assert res.params["n"] == pytest.approx(hill.bounds["n"][1])
    assert any("張り付いています" in w for w in res.warnings)


def test_log_scale_parameters_survive_a_runaway_without_overflowing():
    """Leaving the upper bound of the logarithm at infinity overflows at 10**309."""
    conc = np.array([0.2, 0.32, 0.5, 0.8, 1.1, 1.6])
    signal = langmuir(conc, 9.0, 1.0, 0.0) + np.random.default_rng(0).normal(0, 0.05, conc.size)
    res = fit(conc, signal, fixed={"baseline": 0.0}, ci="profile")
    assert np.isfinite(res.params["kd"])


def test_is_log_scale_is_derived_from_the_bounds_alone():
    custom = Model(
        name="custom",
        params=("k", "a"),
        func=lambda conc, k, a: a * conc / (k + conc),
        bounds={"k": (1e-20, np.inf), "a": (0.0, 10.0)},
        initial=lambda conc, signal: {"k": 1.0, "a": 1.0},
        display={"k": "K", "a": "A"},
        location="k",
        amplitude="a",
        baseline=None,
        description="test model",
    )
    assert custom.is_log_scale("k")
    assert not custom.is_log_scale("a")
