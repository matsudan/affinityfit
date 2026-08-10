"""Required arguments, the display of a zero-width interval, and warm-start independence between directions.

- `model` and `params` of `diagnose` are positional arguments that cannot be omitted.
- A zero-width interval is not printed as a bare number, which would read as uncertainty never assessed.
- The warm start of the profile search is reset for each direction, so the upper limit does not depend on the
  search for the lower one.
"""

from __future__ import annotations

import numpy as np
import pytest

from bindfit import Dataset, diagnose, fit, fit_global, hill, langmuir
from bindfit.uncertainty import Interval, profile_bounds

CONC = np.concatenate([[0.0], np.logspace(-1, 3, 12)])
EXACT = langmuir(CONC, 10.0, 1.0, 0.02)


# -------------------------------------- required arguments of diagnose


def test_diagnose_needs_the_model_and_parameters_positionally():
    """A type checker catches a missing argument at the call site; Python raises TypeError at runtime too,
    which removes the need for the doubled arrangement of a `None` default plus an explicit raise.
    """
    with pytest.raises(TypeError):
        diagnose(CONC, EXACT)  # ty: ignore[missing-argument]
    with pytest.raises(TypeError):
        diagnose(CONC, EXACT, langmuir)  # ty: ignore[missing-argument]


def test_diagnose_works_when_both_are_given():
    messages = diagnose(CONC, EXACT, langmuir, {"kd": 10.0, "bmax": 1.0, "baseline": 0.02})
    assert isinstance(messages, tuple)


def test_diagnose_has_no_none_default_for_params():
    """With the defaults removed, the runtime `params is required` check is no longer needed."""
    import inspect

    signature = inspect.signature(diagnose)
    assert signature.parameters["model"].default is inspect.Parameter.empty
    assert signature.parameters["params"].default is inspect.Parameter.empty


# ------------------------------------- display of a zero-width interval


def test_exact_fit_reports_why_the_interval_is_zero_width():
    """A bare number would read as the uncertainty never having been assessed."""
    res = fit(CONC, EXACT, ci="profile", unit="nM")
    for name in langmuir.params:
        interval = res.intervals[name]
        assert interval.zero_width, name
        text = interval.format("nM" if name == "kd" else "")
        assert "+/- 0" in text
        assert "no residual scatter" in text


def test_zero_width_property():
    assert Interval(point=10.0, lower=10.0, upper=10.0).zero_width
    assert not Interval(point=10.0, lower=9.0, upper=11.0).zero_width
    assert not Interval(point=10.0, lower=None, upper=None).zero_width
    assert not Interval(point=10.0, lower=2.0, upper=None).zero_width


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (10.0, 10.0),
        (9.0, 11.0),
        (None, None),
        (2.0, None),
        (None, 5.0),
        (np.inf, np.inf),
        (-np.inf, -np.inf),
        (np.nan, np.nan),
        (0.0, np.inf),
    ],
)
def test_format_and_zero_width_agree(lower, upper):
    """Writing the test in two places lets one copy drift, so the display must follow `zero_width`."""
    interval = Interval(point=10.0, lower=lower, upper=upper)
    says_zero_width = "no residual scatter" in interval.format()
    assert says_zero_width is interval.zero_width, (lower, upper)


@pytest.mark.parametrize("limit", [np.inf, -np.inf, np.nan])
def test_matching_non_finite_limits_are_not_zero_width(limit):
    """Limits that agree at infinity are not zero width; the limit itself is undetermined."""
    interval = Interval(point=10.0, lower=limit, upper=limit)
    assert not interval.zero_width
    assert not interval.bounded
    assert "undetermined" in interval.format()


def test_zero_width_row_is_not_a_bare_number():
    """The report() line lines up with those of the other parameters."""
    text = fit(CONC, EXACT, ci="profile", unit="nM").report()
    for line in text.splitlines():
        if line.startswith(("Kd", "Bmax", "baseline")):
            assert "+/-" in line, line


def test_noisy_data_still_gets_a_real_interval():
    signal = EXACT + np.random.default_rng(0).normal(0, 0.01, CONC.size)
    res = fit(CONC, signal, ci="profile")
    assert not res.intervals["kd"].zero_width
    assert res.intervals["kd"].half_width > 0


# ------------------------- the warm start is reset for each direction


def test_profile_bounds_resets_before_each_direction():
    """`on_direction_start` is called at the start of each of the two directions."""
    calls: list[str] = []

    def ssr(value):
        calls.append("eval")
        return 1.0 + (value - 10.0) ** 2

    def reset():
        calls.append("reset")

    profile_bounds(
        ssr,
        best=10.0,
        ssr_min=1.0,
        threshold=2.0,
        search_lower=1e-3,
        search_upper=1e3,
        log_scale=True,
        on_direction_start=reset,
    )
    assert calls.count("reset") == 2
    assert calls[0] == "reset"  # called before the first evaluation
    # the second reset comes after the evaluations of the first direction
    assert calls.index("reset", 1) > 1


def test_upper_limit_does_not_depend_on_walking_the_lower_side_first():
    """The upper limit agrees whether it is computed alone or together with the lower one."""

    def ssr(value):
        return 1.0 + (value - 10.0) ** 2

    both = profile_bounds(
        ssr, best=10.0, ssr_min=1.0, threshold=2.0, search_lower=1e-3, search_upper=1e3, log_scale=True
    )
    upper_only = profile_bounds(
        ssr, best=10.0, ssr_min=1.0, threshold=2.0, search_lower=10.0, search_upper=1e3, log_scale=True
    )
    assert both[1] == pytest.approx(upper_only[1])


def test_profile_intervals_are_unchanged_on_well_conditioned_data():
    """Introducing the reset leaves the result unchanged on ordinary data."""
    signal = EXACT + np.random.default_rng(1).normal(0, 0.02, CONC.size)
    interval = fit(CONC, signal, ci="profile").intervals["kd"]
    assert interval.lower == pytest.approx(8.93681, rel=1e-4)
    assert interval.upper == pytest.approx(10.9791, rel=1e-4)


def test_profile_still_reports_one_sided_limits_on_the_hard_case():
    """One-sided limits keep being returned in the ill-conditioned case where the range reaches 0.18 times Kd."""
    conc = np.array([0.2, 0.32, 0.5, 0.8, 1.1, 1.6])
    for seed in range(6):
        signal = langmuir(conc, 9.0, 1.0, 0.0) + np.random.default_rng(seed).normal(0, 0.02, conc.size)
        interval = fit(conc, signal, fixed={"baseline": 0.0}, ci="profile").intervals["kd"]
        assert interval.upper is None, seed
        assert interval.lower is not None and interval.lower > 0, seed


def test_global_fit_profile_intervals_still_bracket_the_truth():
    conc = np.array([0.2, 0.32, 0.5, 0.8, 1.1, 1.6])
    rng = np.random.default_rng(4)
    datasets = [
        Dataset("oxidized", conc, langmuir(conc, 1.1, 1.0, 0.0) + rng.normal(0, 0.01, conc.size)),
        Dataset("reduced", conc, langmuir(conc, 9.0, 1.0, 0.0) + rng.normal(0, 0.01, conc.size)),
    ]
    res = fit_global(datasets, shared=["bmax"], fixed={"baseline": 0.0}, ci="profile")
    assert res.intervals["reduced"]["kd"].contains(9.0)


def test_hill_profile_intervals_survive_the_reset():
    signal = hill(CONC, 10.0, 1.0, 0.0, 3.0) + np.random.default_rng(2).normal(0, 0.02, CONC.size)
    res = fit(CONC, signal, model=hill, fixed={"baseline": 0.0}, ci="profile")
    assert res.intervals["n"].contains(3.0)
    assert res.intervals["kd"].contains(10.0)
