"""The three confidence-interval methods, and rounding that follows the uncertainty.

There are two axes to these checks.
1. On well-sampled data the three methods agree (each one cross-checks the others)
2. On unidentifiable data profile returns a one-sided limit and the asymptotic CI breaks down
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import stats

from affinityfit import Dataset, fit, fit_global, hill, langmuir
from affinityfit.uncertainty import (
    Interval,
    _significant,
    _spans_orders_of_magnitude,
    format_with_uncertainty,
)

# A measured range that reaches only 0.18 times Kd = 9.0 mM: 0.2-1.6 mM.
L_NARROW = np.array([0.2, 0.32, 0.5, 0.8, 1.1, 1.6])
FIXED_BASELINE = {"baseline": 0.0}


def good_data(kd=10.0, noise=0.02, seed=1, points=12):
    """Well-conditioned data whose measured range brackets Kd."""
    conc = np.concatenate([[0.0], np.logspace(np.log10(kd / 100), np.log10(kd * 100), points - 1)])
    signal = langmuir(conc, kd, 1.0, 0.02) + np.random.default_rng(seed).normal(0, noise, conc.size)
    return conc, signal


def narrow_range_data(noise=0.02, seed=0):
    """Data measured only up to 0.18 times Kd."""
    return L_NARROW, langmuir(L_NARROW, 9.0, 1.0, 0.0) + np.random.default_rng(seed).normal(0, noise, L_NARROW.size)


def ssr_with(conc, signal, fixed):
    res = fit_global([Dataset("d", conc, signal)], fixed=fixed)
    resid = res.result_for("d").residuals(conc, signal)
    return float(resid @ resid)


# ---------------------------------------------------------- properties of Interval


def test_interval_bounded_and_half_width():
    iv = Interval(point=10.0, lower=9.0, upper=11.0)
    assert iv.bounded
    assert iv.half_width == pytest.approx(1.0)
    assert iv.symmetric
    assert iv.contains(9.5) and not iv.contains(8.9)


def test_interval_one_sided_has_infinite_half_width():
    iv = Interval(point=10.0, lower=2.0, upper=None)
    assert not iv.bounded
    assert iv.half_width == np.inf
    assert not iv.symmetric
    assert iv.contains(1e9)
    assert not iv.contains(1.0)


# ------------------------------- an infinite limit is not treated as a limit


@pytest.mark.parametrize(
    ("lower", "upper"),
    [(0.0, np.inf), (-np.inf, 1.0), (-np.inf, np.inf), (np.nan, 1.0), (1.0, np.nan)],
)
def test_non_finite_limits_are_not_bounded(lower, upper):
    """The convention established for None must not be slipped past by inf or nan."""
    iv = Interval(point=1.0, lower=lower, upper=upper)
    assert not iv.bounded
    assert iv.half_width == np.inf
    assert not iv.symmetric


def test_infinite_upper_limit_is_reported_as_one_sided():
    iv = Interval(point=1.15e7, lower=3.0, upper=np.inf)
    text = iv.format("nM")
    assert "inf" not in text
    assert text.startswith("> 3")
    assert "undetermined" in text


def test_infinite_limits_do_not_warn():
    """The inf - inf inside symmetric must not raise a RuntimeWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        iv = Interval(point=1.0, lower=0.0, upper=np.inf)
        assert not iv.symmetric
        assert iv.half_width == np.inf
        assert iv.format() is not None


def test_infinite_limit_counts_as_unbounded_for_contains():
    iv = Interval(point=1.0, lower=0.0, upper=np.inf)
    assert iv.contains(1e300)


def test_asymptotic_interval_returns_undetermined_instead_of_infinity():
    """When the curvature is nearly 0, the logarithmic path must not produce inf."""
    conc = np.array([0.2, 0.32, 0.5, 0.8, 1.1, 1.6])
    found = False
    for seed in range(30):
        signal = langmuir(conc, 9.0, 1.0, 0.0) + np.random.default_rng(seed).normal(0, 0.10, conc.size)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            res = fit(conc, signal, fixed={"baseline": 0.0}, ci="asymptotic")
        interval = res.intervals["kd"]
        for limit in (interval.lower, interval.upper):
            assert limit is None or np.isfinite(limit), (seed, limit)
        if not interval.bounded:
            found = True
            assert "limit_undetermined" in {diagnostic.code for diagnostic in res.warnings}, seed
    assert found, "no unidentifiable fit came up in any of the realisations"


def test_interval_asymmetric_is_detected():
    assert not Interval(point=10.0, lower=9.5, upper=30.0).symmetric


def test_interval_format_one_sided():
    assert Interval(point=1e6, lower=0.3, upper=None).format("mM").startswith("> 0.3 mM")
    assert Interval(point=1e-9, lower=None, upper=5.0).format().startswith("< 5")
    assert "both limits undetermined" in Interval(point=1e6, lower=None, upper=None).format()


# --------------------------------------------------- rounding to the significant digits


@pytest.mark.parametrize(
    ("point", "unc", "expected"),
    [
        (4.70e-8, 0.82e-8, "(4.7 +/- 0.8)e-08"),  # a relative error of 17%; should not be reported to 3 digits
        (1.6, 0.2, "1.60 +/- 0.20"),  # uncertainty leads with 2 -> 2 digits
        (0.14, 0.01, "0.140 +/- 0.010"),  # uncertainty leads with 1 -> 3 digits
        (22.0, 5.0, "22 +/- 5"),  # leads with 5 -> 1 digit
        (108.1, 66.6, "110 +/- 70"),
    ],
)
def test_format_with_uncertainty(point, unc, expected):
    assert format_with_uncertainty(point, unc) == expected


def test_format_with_uncertainty_carries_unit():
    assert format_with_uncertainty(1.6, 0.2, "uM") == "1.60 +/- 0.20 uM"


def test_format_with_uncertainty_falls_back_when_unusable():
    assert format_with_uncertainty(1.6, 0.0) == "1.6"
    assert format_with_uncertainty(1.6, np.inf) == "1.6"


# ------------------------------------- displaying an interval that spans orders of magnitude (never printing 0)


@pytest.mark.parametrize(
    ("point", "lower", "upper"),
    [
        (13.0, 0.0312, 5435.0),  # the interval when measurements reach only 0.18 times Kd
        (250.0, 0.4, 9e4),
        (0.6, 0.0009, 400.0),
        (1e-12, 1e-15, 1e-9),
    ],
)
def test_a_positive_limit_is_never_printed_as_zero(point, lower, upper):
    """Kd = 0 means infinite affinity, so it must never be printed.

    Rounding went through a shared decimal place, so at the precision of the narrow side the lower limit
    collapsed to 0.
    """
    text = Interval(point=point, lower=lower, upper=upper).format("nM")
    limits = text.split("[")[1].split("]")[0].split(", ")
    assert float(limits[0]) > 0, text
    assert float(limits[1]) > 0, text


def test_wide_interval_keeps_both_limits_readable():
    text = Interval(point=13.0, lower=0.0312, upper=5435.0).format("nM")
    assert "0.031" in text and "5400" in text
    assert "[0," not in text


def test_narrow_interval_keeps_the_shared_decimal_place():
    """Regression guard. In a nearly symmetric interval the three numbers keep the same decimal place."""
    assert Interval(point=1.076, lower=1.05, upper=1.11).format() == "1.076 [1.050, 1.110] (95% CI)"
    assert Interval(point=108.0, lower=64.0, upper=240.0).format() == "108 [64, 240] (95% CI)"


def test_interval_straddling_zero_is_not_treated_as_wide():
    text = Interval(point=0.01, lower=-0.5, upper=0.6).format()
    assert "-0.50" in text and "0.60" in text


def test_negative_interval_spanning_orders_of_magnitude():
    """Intervals spanning orders of magnitude must work with a negative amplitude too, as in fluorescence quenching."""
    text = Interval(point=-0.8, lower=-100.0, upper=-0.5).format()
    assert "-100" in text and "-0.50" in text


def test_spans_orders_of_magnitude_helper():
    assert _spans_orders_of_magnitude(0.01, 10.0)
    assert _spans_orders_of_magnitude(-10.0, -0.01)
    assert not _spans_orders_of_magnitude(1.0, 2.0)
    assert not _spans_orders_of_magnitude(-1.0, 1.0)  # straddles 0
    assert not _spans_orders_of_magnitude(0.0, 1.0)


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [
        (400.0, 2, "400"),  # plain notation rather than 4e+02
        (5435.0, 2, "5400"),  # rounded to 2 significant digits, then written plainly
        (0.0312, 2, "0.031"),
        (0.0009, 2, "0.00090"),
        (9e4, 2, "90000"),
        (4.7e-8, 2, "4.7e-08"),  # exponent notation once the magnitude is far away
        (0.0, 2, "0"),
    ],
)
def test_significant_formats_plainly_when_readable(value, digits, expected):
    assert _significant(value, digits) == expected


# -------------------------------------------- the three methods cross-checked (well-conditioned data)


def test_three_methods_agree_on_well_conditioned_data():
    conc, signal = good_data()
    got = {m: fit(conc, signal, ci=m, n_boot=400).intervals["kd"] for m in ("asymptotic", "profile", "bootstrap")}
    for iv in got.values():
        assert iv.bounded
        assert iv.contains(10.0)
    # the widths stay within the same order of magnitude
    widths = [iv.half_width for iv in got.values()]
    assert max(widths) / min(widths) < 3.0


def test_profile_endpoints_sit_exactly_on_the_f_test_threshold():
    """Recomputes the definition of the profile interval itself, independently."""
    conc, signal = good_data()
    res = fit(conc, signal, ci="profile")
    ssr_min = ssr_with(conc, signal, {})
    dof = len(conc) - 3
    threshold = ssr_min * (1.0 + float(stats.f.ppf(0.95, 1, dof)) / dof)
    for name in ("kd", "bmax", "baseline"):
        iv = res.intervals[name]
        for edge in (iv.lower, iv.upper):
            assert ssr_with(conc, signal, {name: edge}) == pytest.approx(threshold, rel=1e-4)


def test_profile_interval_contains_the_asymptotic_point():
    conc, signal = good_data()
    res = fit(conc, signal, ci="profile")
    assert res.intervals["kd"].contains(res.params["kd"])


def test_interval_widens_with_noise_for_every_method():
    conc, clean = good_data(noise=0.002, seed=3)
    _, noisy = good_data(noise=0.05, seed=3)
    for method in ("asymptotic", "profile", "bootstrap"):
        narrow = fit(conc, clean, ci=method, n_boot=300).intervals["kd"].half_width
        wide = fit(conc, noisy, ci=method, n_boot=300).intervals["kd"].half_width
        assert wide > narrow, method


# ------------------------------------- unidentifiable data (measured range short of Kd)


def test_asymptotic_interval_breaks_down_when_unidentifiable():
    """The asymptotic CI still returns a finite two-sided interval when the parameter is unidentifiable.

    Kd is estimated on a logarithmic scale, so the lower limit never turns negative, but the interval spreads
    over more than two orders of magnitude and on top of that claims that an upper limit exists. Profile judges
    the upper limit undetermined here (the next test).
    """
    conc, signal = narrow_range_data(noise=0.02, seed=0)
    iv = fit(conc, signal, fixed=FIXED_BASELINE, ci="asymptotic").intervals["kd"]
    assert iv.lower is not None and iv.upper is not None
    assert iv.lower > 0  # a concentration, so it cannot be negative
    assert iv.upper / iv.lower > 100  # even so, it spreads over more than two orders of magnitude
    assert iv.bounded  # and it claims that an upper limit exists


def test_asymptotic_interval_for_a_concentration_is_never_negative():
    """The interval is built on a logarithmic scale, so a concentration constant never gets a negative lower limit."""
    for seed in range(5):
        conc, signal = narrow_range_data(noise=0.05, seed=seed)
        iv = fit(conc, signal, fixed=FIXED_BASELINE, ci="asymptotic").intervals["kd"]
        if iv.lower is not None:
            assert iv.lower > 0, seed


def test_profile_returns_a_one_sided_limit_when_unidentifiable():
    """Profile judges the upper limit "undetermined" and returns the lower limit alone."""
    conc, signal = narrow_range_data(noise=0.02, seed=0)
    iv = fit(conc, signal, fixed=FIXED_BASELINE, ci="profile").intervals["kd"]
    assert iv.upper is None
    assert iv.lower is not None and iv.lower > 0
    assert iv.format("mM").startswith(">")


def test_one_sided_limit_is_stable_across_noise_realisations():
    conc = L_NARROW
    for seed in range(5):
        _, signal = narrow_range_data(noise=0.02, seed=seed)
        iv = fit(conc, signal, fixed=FIXED_BASELINE, ci="profile").intervals["kd"]
        assert iv.upper is None, seed


def test_more_noise_removes_even_the_lower_limit():
    """When the noise is too large for the span of the signal change, not even the lower limit is determined."""
    outcomes = []
    for seed in range(5):
        _, signal = narrow_range_data(noise=0.05, seed=seed)
        iv = fit(L_NARROW, signal, fixed=FIXED_BASELINE, ci="profile").intervals["kd"]
        outcomes.append(iv.lower is None and iv.upper is None)
    assert any(outcomes)


def test_less_noise_restores_both_limits():
    _, signal = narrow_range_data(noise=0.001, seed=0)
    iv = fit(L_NARROW, signal, fixed=FIXED_BASELINE, ci="profile").intervals["kd"]
    assert iv.bounded
    assert iv.contains(9.0)


def test_warns_when_a_limit_is_undetermined():
    conc, signal = narrow_range_data(noise=0.02, seed=0)
    res = fit(conc, signal, fixed=FIXED_BASELINE, ci="profile")
    assert "limit_undetermined" in {diagnostic.code for diagnostic in res.warnings}


def test_sharing_restores_a_two_sided_interval():
    """Sharing the amplitude gives the undersampled form's Kd a two-sided interval."""
    rng = np.random.default_rng(4)
    y_ox = langmuir(L_NARROW, 1.1, 1.0, 0.0) + rng.normal(0, 0.01, L_NARROW.size)
    y_red = langmuir(L_NARROW, 9.0, 1.0, 0.0) + rng.normal(0, 0.01, L_NARROW.size)

    alone = fit(L_NARROW, y_red, fixed=FIXED_BASELINE, ci="profile").intervals["kd"]
    assert alone.upper is None

    together = fit_global(
        [Dataset("oxidized", L_NARROW, y_ox), Dataset("reduced", L_NARROW, y_red)],
        shared=["bmax"],
        fixed=FIXED_BASELINE,
        ci="profile",
    )
    shared_iv = together.intervals["reduced"]["kd"]
    assert shared_iv.bounded
    assert shared_iv.contains(9.0)


# ------------------------------------------------------------ bootstrap


def test_bootstrap_over_replicates_uses_the_replicate_spread():
    conc, _ = good_data()
    rng = np.random.default_rng(9)
    truth = langmuir(conc, 10.0, 1.0, 0.02)
    reps = np.array([truth + rng.normal(0, 0.02, conc.size) for _ in range(3)])
    res = fit(conc, reps.mean(axis=0), ci="bootstrap", replicates=reps, n_boot=400)
    iv = res.intervals["kd"]
    assert iv.bounded and iv.contains(10.0)
    assert res.method == "bootstrap"


def test_bootstrap_is_reproducible_with_a_seed():
    conc, signal = good_data()
    a = fit(conc, signal, ci="bootstrap", n_boot=200, seed=5).intervals["kd"]
    b = fit(conc, signal, ci="bootstrap", n_boot=200, seed=5).intervals["kd"]
    assert (a.lower, a.upper) == (b.lower, b.upper)


def test_bootstrap_distribution_is_skewed_when_unidentifiable():
    conc, signal = narrow_range_data(noise=0.02, seed=0)
    iv = fit(conc, signal, fixed=FIXED_BASELINE, ci="bootstrap", n_boot=400).intervals["kd"]
    assert iv.lower is not None and iv.upper is not None
    # the upper tail is extremely long
    assert (iv.upper - iv.point) / max(iv.point - iv.lower, 1e-12) > 5.0


def test_dataset_validates_replicate_shape():
    conc = np.array([1.0, 2.0, 3.0])
    signal = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="replicates must have shape"):
        Dataset("d", conc, signal, replicates=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="at least 2 rows"):
        Dataset("d", conc, signal, replicates=np.zeros((1, 3)))


# ----------------------------------------------------------- integration and argument passing


def test_unknown_ci_method_is_rejected():
    conc, signal = good_data()
    with pytest.raises(ValueError, match="Unknown ci method"):
        # This checks the runtime guard, so a value that is invalid to the type checker is passed deliberately.
        fit(conc, signal, ci="magic")  # ty: ignore[invalid-argument-type]


def test_method_is_recorded_and_reported():
    conc, signal = good_data()
    for method in ("asymptotic", "profile", "bootstrap"):
        res = fit(conc, signal, ci=method, n_boot=100)
        assert res.method == method
        assert method in res.report()


def test_fixed_parameter_interval_is_degenerate():
    conc, signal = good_data()
    res = fit(conc, signal, fixed={"baseline": 0.0})
    iv = res.intervals["baseline"]
    assert iv.lower == iv.upper == 0.0
    assert res.ci95["baseline"] == 0.0


def test_hill_coefficient_uses_the_interval_for_its_verdict():
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 15)])
    signal = langmuir(conc, 10.0, 1.0, 0.0) + np.random.default_rng(3).normal(0, 0.02, conc.size)
    res = fit(conc, signal, model=hill, ci="profile")
    assert res.intervals["n"].contains(1.0)
    assert "hill_n_includes_one" in {diagnostic.code for diagnostic in res.warnings}


def test_global_fit_reports_intervals_per_dataset():
    # both datasets use the same concentration series (good_data builds its series from kd, so it would not match)
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 12)])
    rng = np.random.default_rng(1)
    signal_a = langmuir(conc, 5.0, 1.0, 0.0) + rng.normal(0, 0.01, conc.size)
    signal_b = langmuir(conc, 50.0, 1.0, 0.0) + rng.normal(0, 0.01, conc.size)
    res = fit_global(
        [Dataset("a", conc, signal_a), Dataset("b", conc, signal_b)],
        shared=["bmax"],
        fixed=FIXED_BASELINE,
        ci="profile",
        unit="nM",
    )
    assert res.intervals["a"]["bmax"] == res.intervals["b"]["bmax"]
    assert res.intervals["a"]["kd"].contains(5.0)
    assert res.intervals["b"]["kd"].contains(50.0)
    assert "profile" in res.report()
