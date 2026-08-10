"""The model plugin mechanism, and the Hill and Michaelis-Menten models."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from affinityfit import MODELS, Dataset, fit, fit_global, hill, langmuir, michaelis
from affinityfit.fitting import _Problem
from affinityfit.models import Model


def hill_data(kd=10.0, bmax=1.0, baseline=0.0, n=2.0, points=12, noise=0.0, seed=0):
    conc = np.concatenate([[0.0], np.logspace(np.log10(kd / 30), np.log10(kd * 30), points - 1)])
    signal = hill(conc, kd, bmax, baseline, n)
    if noise:
        signal = signal + np.random.default_rng(seed).normal(0.0, noise * bmax, size=signal.shape)
    return conc, signal


# ------------------------------------------------------- Basic properties of Model


def test_registry_lists_every_model():
    assert set(MODELS) == {"langmuir", "hill", "michaelis"}
    for name, model in MODELS.items():
        assert isinstance(model, Model)
        assert model.name == name


def test_models_declare_consistent_roles():
    """The diagnostics are written against the role names, so every role must appear in `params`."""
    for model in MODELS.values():
        assert model.location in model.params
        assert model.amplitude in model.params
        assert model.baseline is None or model.baseline in model.params
        assert set(model.bounds) == set(model.params)
        assert set(model.display) == set(model.params)


def test_model_is_callable_as_a_plain_function():
    conc = np.array([0.0, 1.0, 10.0, 100.0])
    np.testing.assert_allclose(langmuir(conc, 10.0, 1.0, 0.0), conc / (10.0 + conc))


def test_model_rejects_wrong_number_of_parameters():
    with pytest.raises(TypeError, match="takes 3 parameters"):
        langmuir(np.array([1.0]), 1.0, 1.0)
    with pytest.raises(TypeError, match="takes 4 parameters"):
        hill(np.array([1.0]), 1.0, 1.0, 0.0)


def test_initial_guess_covers_every_parameter():
    conc, signal = hill_data()
    for model in MODELS.values():
        guess = model.initial(conc, signal)
        assert set(guess) == set(model.params)


def test_ordered_maps_names_to_positional_order():
    values = {"baseline": 0.5, "bmax": 2.0, "kd": 3.0, "n": 1.5}
    assert hill.ordered(values) == (3.0, 2.0, 0.5, 1.5)


# ------------------------------------------------------------------- Hill model


def test_hill_reduces_to_langmuir_when_n_is_one():
    conc = np.array([0.0, 1.0, 5.0, 10.0, 50.0, 200.0])
    np.testing.assert_allclose(
        hill(conc, 10.0, 1.0, 0.02, 1.0),
        langmuir(conc, 10.0, 1.0, 0.02),
        rtol=1e-12,
    )


# ------------------------------- Numerical soundness (never returning NaN)


def test_hill_does_not_return_nan_when_kd_underflows():
    """No NaN in the region where `kd**n` underflows.

    At kd = 1e-30 and n = 20, `kd**n` is 1e-600 and collapses to 0. The naive
    `bmax * conc**n / (kd**n + conc**n)` evaluated 0/0 at the conc = 0 point.
    """
    out = hill(np.array([0.0, 1.0]), 1e-30, 1.0, 0.0, 20.0)
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0  # baseline
    assert out[1] == pytest.approx(1.0)


def test_hill_is_finite_across_the_whole_parameter_domain():
    """Inside the declared bounds the objective function is finite everywhere."""
    conc = np.array([0.0, 1e-18, 1e-12, 1e-6, 1.0, 1e6, 1e12])
    for exponent in range(-30, 7):
        for n in (0.05, 0.5, 1.0, 2.0, 7.0, 13.0, 20.0):
            for baseline in (0.0, -1.0, 2.5):
                out = hill(conc, 10.0**exponent, 1.0, baseline, n)
                assert np.all(np.isfinite(out)), (exponent, n, baseline)


def test_hill_emits_no_runtime_warning_in_extreme_regions():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        hill(np.array([0.0, 1e-20, 1.0, 1e20]), 1e-30, 1.0, 0.0, 20.0)
        hill(np.array([0.0, 1e-20, 1.0, 1e20]), 1e30, 1.0, 0.0, 20.0)


def test_hill_residuals_are_finite_at_the_corners_of_the_bounds():
    """The residuals stay finite at the combinations the optimiser can try as trial points."""
    conc = np.concatenate([[0.0], np.logspace(-15, -13, 9)])
    problem = _Problem([Dataset("d", conc, np.zeros_like(conc))], hill, (), {"baseline": 0.0})
    for kd in (1e-30, 1e-20, 1e-13, 1.0, 1e10):
        for n in (0.05, 1.0, 20.0):
            x = problem.x0.copy()
            x[problem.slot_of("kd", 0)] = kd
            x[problem.slot_of("n", 0)] = n
            assert np.all(np.isfinite(problem.residual(x))), (kd, n)


def test_hill_matches_the_direct_formula_in_the_ordinary_range():
    """The rewritten form is algebraically equivalent to the direct one."""

    def direct(conc, kd, bmax, baseline, n):
        powered = np.power(conc, n)
        return baseline + bmax * powered / (kd**n + powered)

    for kd in (1e-9, 1e-3, 1.0, 50.0):
        for n in (0.3, 1.0, 2.5, 6.0):
            conc = np.concatenate([[0.0], np.logspace(np.log10(kd / 100), np.log10(kd * 100), 40)])
            np.testing.assert_allclose(
                hill(conc, kd, 1.0, 0.02, n),
                direct(conc, kd, 1.0, 0.02, n),
                atol=1e-15,
            )


def test_hill_limits_are_exact():
    assert hill(np.array([10.0]), 10.0, 1.0, 0.0, 3.0)[0] == pytest.approx(0.5)
    assert hill(np.array([0.0]), 10.0, 1.0, 0.25, 3.0)[0] == pytest.approx(0.25)
    assert hill(np.array([1e12]), 10.0, 1.0, 0.0, 3.0)[0] == pytest.approx(1.0)


def test_hill_handles_a_negative_amplitude_at_extreme_kd():
    """Fluorescence quenching (bmax < 0) combined with an extreme kd."""
    out = hill(np.array([0.0, 1.0, 1e6]), 1e-25, -0.8, 1.0, 18.0)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(1.0)
    assert out[-1] == pytest.approx(0.2)


def test_hill_half_saturation_is_at_kd_for_any_n():
    for n in (0.5, 1.0, 2.0, 4.0):
        assert hill(np.array([10.0]), 10.0, 1.0, 0.0, n)[0] == pytest.approx(0.5)


def test_hill_recovers_known_coefficient():
    conc, signal = hill_data(kd=10.0, bmax=1.0, baseline=0.0, n=2.5)
    res = fit(conc, signal, model=hill)
    assert res.params["n"] == pytest.approx(2.5, rel=1e-3)
    assert res.params["kd"] == pytest.approx(10.0, rel=1e-3)


def test_hill_recovers_coefficient_with_noise():
    conc, signal = hill_data(kd=10.0, n=2.0, points=16, noise=0.02, seed=7)
    res = fit(conc, signal, model=hill)
    assert res.params["n"] == pytest.approx(2.0, rel=0.2)
    assert res.params["n"] - res.ci95["n"] < 2.0 < res.params["n"] + res.ci95["n"]


def test_hill_on_noncooperative_data_warns_that_n_includes_one():
    """When the data are 1:1, the fit says that cooperativity cannot be claimed."""
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 15)])
    signal = langmuir(conc, 10.0, 1.0, 0.0) + np.random.default_rng(3).normal(0, 0.02, conc.size)
    res = fit(conc, signal, model=hill)
    assert any("信頼区間が 1 を含みます" in w for w in res.warnings)


def test_hill_warns_when_n_is_significantly_below_one():
    conc, signal = hill_data(kd=10.0, n=0.4, points=20)
    res = fit(conc, signal, model=hill)
    assert any("有意に 1 を下回っています" in w for w in res.warnings)


def test_no_hill_warning_for_langmuir_model():
    conc, signal = hill_data(n=1.0)
    res = fit(conc, signal, model=langmuir)
    assert not any("Hill" in w for w in res.warnings)


def test_aicc_prefers_hill_on_cooperative_data():
    """Checked over several seeds. A claim about model selection cannot rest on a single noise realisation."""
    hits = 0
    for seed in range(20):
        conc, signal = hill_data(kd=10.0, n=3.0, points=16, noise=0.01, seed=seed)
        ds = [Dataset("d", conc, signal)]
        hits += fit_global(ds, model=hill, ci="asymptotic").aicc < fit_global(ds, model=langmuir, ci="asymptotic").aicc
    assert hits == 20, hits


def test_aicc_prefers_langmuir_on_noncooperative_data():
    """On genuinely non-cooperative data, hill with its extra parameter is not the one chosen.

    Criteria of the AIC family do not rule out overfitting entirely, so the rate is not 100%. The
    threshold comes from the selection rate over 60 noise realisations (93% with AICc).
    """
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 15)])
    hits = 0
    trials = 60
    for seed in range(trials):
        signal = langmuir(conc, 10.0, 1.0, 0.0) + np.random.default_rng(seed).normal(0, 0.02, conc.size)
        ds = [Dataset("d", conc, signal)]
        hits += fit_global(ds, model=langmuir, ci="asymptotic").aicc < fit_global(ds, model=hill, ci="asymptotic").aicc
    assert hits >= 0.9 * trials, hits


def test_aicc_overfits_less_often_than_plain_aic():
    """Measures the effect of the correction itself: AICc overfits less often than plain AIC."""
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 15)])
    aic_correct = aicc_correct = 0
    for seed in range(60):
        signal = langmuir(conc, 10.0, 1.0, 0.0) + np.random.default_rng(seed).normal(0, 0.02, conc.size)
        ds = [Dataset("d", conc, signal)]
        simple = fit_global(ds, model=langmuir, ci="asymptotic")
        complex_ = fit_global(ds, model=hill, ci="asymptotic")
        aic_correct += simple.aic < complex_.aic
        aicc_correct += simple.aicc < complex_.aicc
    assert aicc_correct > aic_correct, (aic_correct, aicc_correct)


def test_aicc_penalises_the_extra_hill_parameter_more_than_aic():
    """On the same data, the correction term is larger for hill, which carries one parameter more."""
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 11)])
    signal = langmuir(conc, 10.0, 1.0, 0.0)
    ds = [Dataset("d", conc, signal)]
    lang = fit_global(ds, model=langmuir, ci="asymptotic")
    hill_fit = fit_global(ds, model=hill, ci="asymptotic")
    assert hill_fit.aicc - hill_fit.aic > lang.aicc - lang.aic


def test_hill_curve_and_report_use_the_app_label():
    conc, signal = hill_data(n=2.0)
    res = fit(conc, signal, model=hill, unit="nM")
    text = res.report()
    assert "Kd(app)" in text and "n (Hill)" in text and "hill" in text
    x, y = res.curve()
    assert len(x) == 300
    assert np.all(np.diff(y) > 0)


# --------------------------------------------------- Michaelis-Menten model


def test_michaelis_shares_the_algebra_of_langmuir():
    conc = np.array([0.0, 0.5, 1.0, 2.0, 8.0])
    np.testing.assert_allclose(michaelis(conc, 1.6, 0.014, 0.0), langmuir(conc, 1.6, 0.014, 0.0))


def test_michaelis_recovers_known_km_and_kcat():
    """Recovers known Km and kcat for a catalyst assayed over a typical substrate range.

    Conditions: catalyst 0.1 uM, substrate 0.5-8 uM over 5 points
    True values: Km = 1.6 +/- 0.2 uM, kcat = 0.14 +/- 0.01 min^-1
    """
    enzyme, km, kcat = 0.1, 1.6, 0.14
    substrate = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    velocity = michaelis(substrate, km, kcat * enzyme, 0.0)
    res = fit(substrate, velocity, model=michaelis, fixed={"baseline": 0.0}, unit="uM")
    assert res.params["km"] == pytest.approx(km, rel=1e-4)
    assert res.params["vmax"] / enzyme == pytest.approx(kcat, rel=1e-4)
    assert "Km" in res.report() and "Vmax" in res.report()


def test_michaelis_diagnostics_speak_of_km_not_kd():
    substrate = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    res = fit(substrate, michaelis(substrate, 1.6, 0.014, 0.0), model=michaelis)
    assert res.warnings
    assert all("Kd" not in w for w in res.warnings)
    assert any("Km" in w for w in res.warnings)


# ------------------------------------------------- Combined with a global fit


def test_global_fit_works_with_hill_and_shared_amplitude():
    conc = np.concatenate([[0.0], np.logspace(-1, 2, 11)])
    ds = [
        Dataset("weak", conc, hill(conc, 20.0, 1.0, 0.0, 2.0)),
        Dataset("strong", conc, hill(conc, 2.0, 1.0, 0.0, 2.0)),
    ]
    res = fit_global(ds, model=hill, shared=["bmax", "n"], fixed={"baseline": 0.0})
    # bmax x1 + n x1 + kd x2 = 4
    assert res.n_free_params == 4
    assert res.params["weak"]["n"] == res.params["strong"]["n"]
    assert res.params["weak"]["kd"] == pytest.approx(20.0, rel=1e-3)
    assert res.params["strong"]["kd"] == pytest.approx(2.0, rel=1e-3)


def test_global_fit_rejects_parameter_absent_from_the_model():
    conc, signal = hill_data()
    ds = [Dataset("a", conc, signal), Dataset("b", conc, signal)]
    with pytest.raises(ValueError, match="Model 'langmuir' has"):
        fit_global(ds, model=langmuir, shared=["n"])


def test_result_for_carries_the_model():
    conc, signal = hill_data(n=2.0)
    res = fit_global([Dataset("a", conc, signal)], model=hill)
    sub = res.result_for("a")
    assert sub.model is hill
    assert set(sub.params) == set(hill.params)
