"""Behaviour when the inner optimisation fails.

Profile likelihood pins the parameter of interest at extreme values and refits everything else there, so
non-convergence is easy to hit. Bootstrap runs many resamples, so some of them fail. The former must not bring
the whole fit down; the latter must report how many failed.
"""

from __future__ import annotations

import numpy as np
import pytest

from affinityfit import fit, fitting, langmuir
from affinityfit.uncertainty import MIN_BOOTSTRAP_SAMPLES, profile_bounds

CONC = np.concatenate([[0.0], np.logspace(-1, 3, 12)])
SIGNAL = langmuir(CONC, 10.0, 1.0, 0.02) + np.random.default_rng(0).normal(0, 0.01, CONC.size)


@pytest.fixture
def failing_solve(monkeypatch):
    """Factory for making `_Problem.solve` fail under a given condition."""
    original = fitting._Problem.solve

    def install(should_fail):
        counter = {"n": 0}

        def patched(self, *args, **kwargs):
            counter["n"] += 1
            if should_fail(counter["n"], kwargs):
                raise RuntimeError("Optimization did not converge: simulated")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(fitting._Problem, "solve", patched)
        return counter

    return install


# ------------------------------------- non-convergence during the profile search


def test_profile_survives_a_single_inner_failure(failing_solve):
    """A single inner failure must not bring the whole fit() down."""
    failing_solve(lambda n, kwargs: n == 5)
    res = fit(CONC, SIGNAL, ci="profile")
    assert np.isfinite(res.params["kd"])


def test_profile_failure_widens_rather_than_narrows_the_interval(failing_solve):
    """When the search can no longer narrow the interval, err wide. Erring narrow is the dangerous direction."""
    clean = fit(CONC, SIGNAL, ci="profile").intervals["kd"]
    failing_solve(lambda n, kwargs: n == 5)
    degraded = fit(CONC, SIGNAL, ci="profile").intervals["kd"]
    assert clean.lower is not None and degraded.lower is not None
    assert clean.upper is not None and degraded.upper is not None
    assert degraded.lower <= clean.lower
    assert degraded.upper >= clean.upper
    assert degraded.lower < degraded.point  # has not collapsed onto the point estimate


def test_profile_reports_undetermined_when_every_inner_fit_fails(failing_solve):
    failing_solve(lambda n, kwargs: bool(kwargs.get("pinned")))
    res = fit(CONC, SIGNAL, ci="profile")
    for name in ("kd", "bmax", "baseline"):
        assert not res.intervals[name].bounded, name
    assert any("片側が決定できない" in w for w in res.warnings)


def test_profile_bounds_treats_none_as_unevaluable_not_as_a_crossing():
    """Substituting inf for None would mimic a threshold crossing and claim a limit that does not exist."""
    lower, upper = profile_bounds(
        lambda value: None,
        best=10.0,
        ssr_min=1.0,
        threshold=2.0,
        search_lower=1e-3,
        search_upper=1e3,
        log_scale=True,
    )
    assert lower is None and upper is None


def test_profile_bounds_still_finds_a_crossing_when_evaluation_works():
    def ssr(value):
        return 1.0 + (value - 10.0) ** 2

    lower, upper = profile_bounds(
        ssr, best=10.0, ssr_min=1.0, threshold=2.0, search_lower=1e-3, search_upper=1e3, log_scale=True
    )
    assert lower == pytest.approx(9.0, rel=1e-3)
    assert upper == pytest.approx(11.0, rel=1e-3)


# --------------------------------------- reporting bootstrap failures


def test_bootstrap_reports_the_failure_count(failing_solve):
    """Even a handful of failures must be reported, since the interval can come out too narrow."""
    failing_solve(lambda n, kwargs: n > 1 and n % 25 == 0)
    res = fit(CONC, SIGNAL, ci="bootstrap", n_boot=400)
    message = next(w for w in res.warnings if "ブートストラップ" in w)
    assert "収束しませんでした" in message
    assert "狭い" in message
    assert res.intervals["kd"].bounded  # the interval is still returned when enough resamples succeeded


def test_bootstrap_refuses_to_report_when_most_resamples_fail(failing_solve):
    """With only 20 of 400 resamples succeeding, no interval must be formed."""
    failing_solve(lambda n, kwargs: n > 1 and n % 20 != 0)
    res = fit(CONC, SIGNAL, ci="bootstrap", n_boot=400)
    assert not res.intervals["kd"].bounded
    message = next(w for w in res.warnings if "ブートストラップ" in w)
    assert "決定不能" in message
    assert str(MIN_BOOTSTRAP_SAMPLES) in message


def test_bootstrap_failure_message_names_the_counts(failing_solve):
    failing_solve(lambda n, kwargs: n > 1 and n % 20 != 0)
    res = fit(CONC, SIGNAL, ci="bootstrap", n_boot=400)
    message = next(w for w in res.warnings if "ブートストラップ" in w)
    assert "400 回" in message and "95%" in message


def test_bootstrap_without_failures_says_nothing_about_convergence():
    res = fit(CONC, SIGNAL, ci="bootstrap", n_boot=200)
    assert not any("ブートストラップ" in w for w in res.warnings)
    assert res.intervals["kd"].bounded


def test_percentile_interval_needs_the_documented_minimum():
    from affinityfit.uncertainty import percentile_interval

    just_under = percentile_interval(np.arange(MIN_BOOTSTRAP_SAMPLES - 1, dtype=float), point=50.0)
    assert not just_under.bounded
    enough = percentile_interval(np.arange(MIN_BOOTSTRAP_SAMPLES, dtype=float), point=50.0)
    assert enough.bounded
