"""Raw statistics and p-values behind the residual-shape and heteroscedasticity checks.

The warnings themselves are thresholded at a fixed significance level, which does not
compose across several fits: reporting several datasets calls for a multiple-comparison
correction chosen by whoever is running the study, not a fixed one baked into the library.
`Statistic` exists so that correction can be done in the caller's own code.
"""

from __future__ import annotations

import numpy as np
import pytest

from bindfit import Dataset, Statistic, fit, fit_global, hill, langmuir

CONC = np.concatenate([[0.0], np.logspace(-1, 3, 15)])


def cooperative(n=3.0, noise=0.02, seed=0):
    """Genuinely cooperative data. Fitting it with a 1:1 model deviates systematically."""
    return CONC, hill(CONC, 10.0, 1.0, 0.0, n) + np.random.default_rng(seed).normal(0, noise, CONC.size)


# ----------------------------------------------------- FitResult.statistics


def test_fit_result_carries_all_three_statistics_when_the_checks_apply():
    conc, signal = cooperative(n=3.0)
    res = fit(conc, signal, model=langmuir)
    names = {s.name for s in res.statistics}
    assert names == {"residual_runs", "residual_autocorrelation", "heteroscedasticity"}


def test_statistic_values_match_the_ones_quoted_in_the_warning():
    """The numbers handed out programmatically must be the same ones printed in the warning text."""
    conc, signal = cooperative(n=3.0)
    res = fit(conc, signal, model=langmuir)
    message = next(w for w in res.warnings if "残差" in w)
    runs_z = next(s for s in res.statistics if s.name == "residual_runs")
    autocorr = next(s for s in res.statistics if s.name == "residual_autocorrelation")
    assert f"z = {runs_z.statistic:.2f}" in message
    assert f"自己相関 = {autocorr.statistic:.2f}" in message


def test_residual_runs_p_value_is_the_one_sided_lower_tail():
    """Only a deficit of runs is evidence of systematic structure, so the tail is one-sided."""
    from scipy import stats as scipy_stats

    conc, signal = cooperative(n=3.0)
    res = fit(conc, signal, model=langmuir)
    runs = next(s for s in res.statistics if s.name == "residual_runs")
    assert runs.p_value == pytest.approx(float(scipy_stats.norm.cdf(runs.statistic)))


def test_autocorrelation_has_no_p_value():
    """Judged against a fixed threshold rather than a null distribution."""
    conc, signal = cooperative(n=3.0)
    res = fit(conc, signal, model=langmuir)
    autocorr = next(s for s in res.statistics if s.name == "residual_autocorrelation")
    assert autocorr.p_value is None
    assert autocorr.alpha == 0.3


def test_no_statistics_when_the_checks_do_not_apply():
    """Fewer than 8 points: the residual-shape checks are skipped entirely, statistics included."""
    conc = np.array([0.0, 1.0, 3.0, 10.0, 30.0])
    res = fit(conc, langmuir(conc, 10.0, 1.0, 0.0) + 0.3)
    assert res.statistics == ()


def test_statistics_are_reported_even_when_no_warning_fires():
    """A clean fit still returns the statistics; only the verdict, not the measurement, is suppressed."""
    signal = langmuir(CONC, 10.0, 1.0, 0.02) + np.random.default_rng(4).normal(0, 0.01, CONC.size)
    res = fit(CONC, signal, model=langmuir)
    assert res.warnings == ()
    assert {s.name for s in res.statistics} == {"residual_runs", "residual_autocorrelation", "heteroscedasticity"}


def test_weighted_fit_omits_the_heteroscedasticity_statistic():
    """Passing sigma suppresses the heteroscedasticity check itself, statistic included."""
    conc, signal = cooperative(n=3.0)
    res = fit(conc, signal, model=langmuir, sigma=np.full_like(signal, 0.02))
    assert not any(s.name == "heteroscedasticity" for s in res.statistics)
    assert any(s.name == "residual_runs" for s in res.statistics)


# --------------------------------------------- degenerate case: every residual one sign


def test_all_residuals_one_sign_reports_a_sign_test_instead_of_runs():
    """The runs count is degenerate (always 1) when every residual shares a sign, so a sign test stands in."""
    params = {"kd": 10.0, "bmax": 1.0, "baseline": 0.0}
    signal = langmuir(CONC, *langmuir.ordered(params)) + 0.5
    from bindfit.core import _residual_structure

    collected: list[Statistic] = []
    msgs = _residual_structure(CONC, signal, langmuir, params, collected)
    assert msgs and "すべてが同じ符号" in msgs[0][1]
    sign_test = next(s for s in collected if s.name == "residual_sign_test")
    assert sign_test.statistic == float(np.count_nonzero(np.sign(signal - langmuir(CONC, *langmuir.ordered(params)))))
    assert sign_test.p_value == pytest.approx(2.0 * 0.5**sign_test.statistic)
    assert not any(s.name == "residual_runs" for s in collected)


# ------------------------------------------------------ heteroscedasticity sign


def test_heteroscedasticity_p_value_flips_when_the_error_shrinks_instead():
    """rho < 0 means the noise shrinks with the fitted value, the opposite of what weighting addresses."""
    rng = np.random.default_rng(0)
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 20)])
    fitted = langmuir(conc, 10.0, 1.0, 0.0)
    sigma_desc = 0.3 - 0.28 * (fitted - fitted.min()) / (fitted.max() - fitted.min())
    signal = fitted + rng.normal(0, sigma_desc)
    res = fit(conc, signal)
    het = next(s for s in res.statistics if s.name == "heteroscedasticity")
    assert het.statistic < 0
    assert het.p_value is not None and het.p_value > 0.5
    assert not any("順位相関" in w for w in res.warnings)


# -------------------------------------------------- GlobalFitResult.statistics_per


def test_global_fit_keeps_statistics_separate_per_dataset():
    rng = np.random.default_rng(1)
    conc = CONC
    datasets = [
        Dataset(f"sample_{i}", conc, hill(conc, 10.0, 1.0, 0.0, n) + rng.normal(0, 0.02, conc.size))
        for i, n in enumerate([1.0, 1.0, 3.0])
    ]
    res = fit_global(datasets, model=langmuir)
    assert set(res.statistics_per) == {"sample_0", "sample_1", "sample_2"}
    for name in res.statistics_per:
        assert {s.name for s in res.statistics_per[name]} == {
            "residual_runs",
            "residual_autocorrelation",
            "heteroscedasticity",
        }
    # the cooperative dataset is the one whose runs test actually rejects
    runs_p = {
        name: next(s.p_value for s in stats if s.name == "residual_runs") for name, stats in res.statistics_per.items()
    }
    assert runs_p["sample_2"] is not None and runs_p["sample_2"] < 0.05
    assert runs_p["sample_0"] is not None and runs_p["sample_0"] > 0.05


def test_result_for_carries_the_per_dataset_statistics():
    rng = np.random.default_rng(1)
    conc = CONC
    datasets = [
        Dataset(f"sample_{i}", conc, hill(conc, 10.0, 1.0, 0.0, n) + rng.normal(0, 0.02, conc.size))
        for i, n in enumerate([1.0, 3.0])
    ]
    res = fit_global(datasets, model=langmuir)
    sub = res.result_for("sample_1")
    assert sub.statistics == res.statistics_per["sample_1"]


# ------------------------------------------------- applying a correction downstream


def test_p_values_collected_across_datasets_can_be_corrected_with_scipy():
    """The intended usage: gather p-values from several fits and hand them to a correction of your choice."""
    scipy_stats = pytest.importorskip("scipy.stats")
    if not hasattr(scipy_stats, "false_discovery_control"):
        pytest.skip("scipy is too old for false_discovery_control")

    rng = np.random.default_rng(2)
    conc = CONC
    datasets = [
        Dataset(f"sample_{i}", conc, hill(conc, 10.0, 1.0, 0.0, n) + rng.normal(0, 0.02, conc.size))
        for i, n in enumerate([1.0, 1.0, 1.0, 3.0])
    ]
    res = fit_global(datasets, model=langmuir)

    names, p_values = [], []
    for name, stats in res.statistics_per.items():
        p = next(s.p_value for s in stats if s.name == "residual_runs")
        names.append(name)
        p_values.append(p)

    q_values = scipy_stats.false_discovery_control(p_values, method="bh")
    assert len(q_values) == len(p_values)
    assert all(q >= p for q, p in zip(q_values, p_values, strict=True))
    # the corrected p-value for the genuinely cooperative dataset should still be the smallest
    assert names[int(np.argmin(q_values))] == "sample_3"


# --------------------------------------------------------------- Statistic itself


def test_statistic_is_frozen():
    """`Statistic` must not be mutable after construction, matching the other result objects in the library."""
    s = Statistic(name="residual_runs", statistic=-2.0, p_value=0.02, alpha=0.025)
    with pytest.raises(AttributeError):
        s.statistic = 0.0  # ty: ignore[invalid-assignment]
