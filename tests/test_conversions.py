"""The Cheng-Prusoff correction from a displacement IC50 to Ki."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from affinityfit import Interval, fit, ic50, ki_from_ic50


def fitted(hillslope: float, seed: int = 0):
    """A dose-response fit with a known slope, noisy enough to have an interval."""
    conc = np.logspace(0, 4, 16)
    clean = ic50(conc, 50.0, -100.0, 100.0, hillslope)
    signal = clean + np.random.default_rng(seed).normal(0.0, 1.0, conc.size)
    return fit(conc, signal, model=ic50, unit="nM")


def user_warnings(call):
    """Every UserWarning a call emits, so a test can assert on their absence as well."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    return [w for w in caught if issubclass(w.category, UserWarning)]


# --------------------------------------------------------------- the correction itself


def test_ki_divides_out_the_tracer_occupancy():
    # 1 + 5/2 = 3.5
    assert ki_from_ic50(50.0, tracer_conc=5.0, tracer_kd=2.0) == pytest.approx(50.0 / 3.5)


def test_ki_equals_ic50_when_nothing_competes():
    assert ki_from_ic50(50.0, tracer_conc=0.0, tracer_kd=2.0) == 50.0


def test_the_correction_grows_with_the_tracer_concentration():
    """The reason the correction exists: more tracer to displace inflates the raw IC50."""
    low = ki_from_ic50(50.0, tracer_conc=1.0, tracer_kd=2.0)
    high = ki_from_ic50(50.0, tracer_conc=20.0, tracer_kd=2.0)
    assert high < low < 50.0


def test_ki_maps_an_interval_exactly_when_the_tracer_constant_is_exact():
    """With nothing else to fold in, dividing by a constant carries the interval across unchanged in shape."""
    interval = Interval(point=50.0, lower=40.0, upper=65.0, method="profile")
    out = ki_from_ic50(interval, tracer_conc=5.0, tracer_kd=2.0)
    assert out.lower is not None
    assert out.upper is not None
    assert out.point == pytest.approx(50.0 / 3.5)
    assert out.lower == pytest.approx(40.0 / 3.5)
    assert out.upper == pytest.approx(65.0 / 3.5)
    assert out.method == "profile"
    # How precise the estimate is comes from the fit, not from the correction, so it is preserved.
    assert (out.upper - out.lower) / out.point == pytest.approx((65.0 - 40.0) / 50.0)


def test_ki_keeps_an_undetermined_limit_undetermined():
    """A one-sided IC50 cannot turn into a two-sided Ki; inventing the missing limit would be the bug."""
    out = ki_from_ic50(Interval(point=50.0, lower=40.0, upper=None), tracer_conc=5.0, tracer_kd=2.0)
    assert out.upper is None
    assert out.lower == pytest.approx(40.0 / 3.5)
    assert not out.bounded


def test_ki_rejects_an_unusable_tracer_kd():
    with pytest.raises(ValueError, match="tracer_kd must be finite and positive"):
        ki_from_ic50(50.0, tracer_conc=5.0, tracer_kd=0.0)


def test_ki_rejects_a_negative_tracer_concentration():
    with pytest.raises(ValueError, match="tracer_conc must be finite and non-negative"):
        ki_from_ic50(50.0, tracer_conc=-1.0, tracer_kd=2.0)


def test_ki_flows_from_a_fitted_ic50():
    """End to end: fit a displacement curve, then report the constant that does not depend on the tracer."""
    conc = np.logspace(0, 4, 16)
    response = ic50(conc, 50.0, -100.0, 100.0, 1.0)
    res = fit(conc, response, model=ic50, unit="nM")
    ki = ki_from_ic50(res.intervals["ic50"], tracer_conc=5.0, tracer_kd=2.0)
    assert ki.point == pytest.approx(50.0 / 3.5, rel=1e-6)


# ------------------------------------------------------------ the slope the form assumes


def test_a_slope_away_from_one_leaves_the_standard_form_behind():
    """The standard form assumes a slope of 1, and a slope whose interval excludes 1 breaks it.

    This is the case that used to pass silently: a 4PL curve of slope 2.5 gives a Ki as
    readily as a slope of 1 does, and nothing said that the equation no longer applied.
    """
    res = fitted(2.5)
    assert not res.intervals["hillslope"].contains(1.0)
    with pytest.warns(UserWarning, match="1 を含みません"):
        ki = ki_from_ic50(res, tracer_conc=5.0, tracer_kd=2.0)
    # The value still comes back: the modified forms disagree with one another, so choosing
    # between them belongs to the caller rather than to this library.
    assert ki.point == pytest.approx(res.params["ic50"] / 3.5, rel=1e-9)


def test_a_slope_consistent_with_one_passes_without_comment():
    res = fitted(1.0, seed=1)
    assert res.intervals["hillslope"].contains(1.0)
    assert user_warnings(lambda: ki_from_ic50(res, tracer_conc=5.0, tracer_kd=2.0)) == []


def test_the_slope_cannot_be_checked_from_an_interval_alone():
    """Handing over one interval hides the slope, which is the reason the result overload exists."""
    res = fitted(2.5)
    assert not res.intervals["hillslope"].contains(1.0)
    assert user_warnings(lambda: ki_from_ic50(res.intervals["ic50"], tracer_conc=5.0, tracer_kd=2.0)) == []


def test_a_fit_result_is_read_through_the_location_role():
    """The half-maximal concentration is found by role, so the overload is not tied to one model."""
    res = fitted(1.0, seed=1)
    assert ki_from_ic50(res, tracer_conc=5.0, tracer_kd=2.0) == ki_from_ic50(
        res.intervals["ic50"], tracer_conc=5.0, tracer_kd=2.0
    )


# ------------------------------------------------ the tracer constant's own uncertainty


def test_the_tracer_constant_carries_its_own_uncertainty_into_ki():
    """Treating Kd* as exact reports a Ki interval narrower than the experiment supports.

    At [T] = 2.5 * Kd the tracer term reaches Ki damped by r/(1+r) = 0.71, so 20% on the
    tracer constant becomes 14% on Ki against roughly 5% from the IC50 here. Leaving it
    out understates the width by a factor of three.
    """
    measured = fitted(1.0, seed=1).intervals["ic50"]
    exact = ki_from_ic50(measured, tracer_conc=5.0, tracer_kd=2.0)
    with_error = ki_from_ic50(measured, tracer_conc=5.0, tracer_kd=Interval(point=2.0, lower=1.6, upper=2.4))
    assert with_error.point == pytest.approx(exact.point)
    assert with_error.half_width > 3 * exact.half_width


def test_the_tracer_term_is_combined_in_quadrature():
    """The two errors come from separate experiments, so they add in quadrature and not linearly."""
    measured = Interval(point=50.0, lower=45.0, upper=55.0)  # 10% either side
    tracer = Interval(point=2.0, lower=1.6, upper=2.4)  # 20%
    out = ki_from_ic50(measured, tracer_conc=5.0, tracer_kd=tracer)

    ratio = 5.0 / 2.0
    expected = math.hypot(0.10, (ratio / (1.0 + ratio)) * 0.20)
    assert out.lower is not None
    assert out.upper is not None
    assert (out.upper - out.point) / out.point == pytest.approx(expected)
    assert (out.point - out.lower) / out.point == pytest.approx(expected)


def test_the_tracer_term_is_damped_when_the_tracer_sits_below_its_kd():
    """r/(1+r) is what keeps a tracer run well under its own Kd from mattering."""
    measured = Interval(point=50.0, lower=45.0, upper=55.0)
    # 20% on the constant, but [T]/Kd is only 0.05, so almost none of it carries through.
    tracer = Interval(point=100.0, lower=80.0, upper=120.0)
    damped = ki_from_ic50(measured, tracer_conc=5.0, tracer_kd=tracer)
    exact = ki_from_ic50(measured, tracer_conc=5.0, tracer_kd=100.0)
    assert damped.half_width / exact.half_width < 1.01


def test_an_asymmetric_interval_stays_asymmetric_through_the_tracer_term():
    measured = Interval(point=50.0, lower=40.0, upper=70.0, method="profile")
    tracer = Interval(point=2.0, lower=1.8, upper=2.2)
    out = ki_from_ic50(measured, tracer_conc=5.0, tracer_kd=tracer)
    assert out.lower is not None
    assert out.upper is not None
    assert (out.upper - out.point) > (out.point - out.lower)
    assert out.method == "profile"


def test_a_lower_limit_swamped_by_the_combined_error_is_undetermined():
    """Ki = 0 reads as infinitely tight binding, so a limit driven past zero is not reported as one."""
    measured = Interval(point=50.0, lower=5.0, upper=200.0)
    tracer = Interval(point=2.0, lower=0.2, upper=3.8)
    out = ki_from_ic50(measured, tracer_conc=5.0, tracer_kd=tracer)
    assert out.lower is None
    assert out.upper is not None
    assert not out.bounded


def test_an_unbounded_tracer_interval_is_refused():
    """With no width to propagate, carrying on would report a precision the tracer never had."""
    with pytest.raises(ValueError, match="undetermined limit"):
        ki_from_ic50(50.0, tracer_conc=5.0, tracer_kd=Interval(point=2.0, lower=1.6, upper=None))


def test_an_exact_ic50_with_an_uncertain_tracer_still_returns_an_interval():
    """The tracer constant on its own is enough spread to report."""
    out = ki_from_ic50(50.0, tracer_conc=5.0, tracer_kd=Interval(point=2.0, lower=1.6, upper=2.4))
    assert isinstance(out, Interval)
    ratio = 5.0 / 2.0
    assert out.half_width / out.point == pytest.approx((ratio / (1.0 + ratio)) * 0.20)


def test_an_exact_pair_still_returns_a_plain_number():
    """The simplest call keeps its simplest answer."""
    assert isinstance(ki_from_ic50(50.0, tracer_conc=5.0, tracer_kd=2.0), float)
