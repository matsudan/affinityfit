"""The model plugin mechanism, and the Hill and Michaelis-Menten models."""

from __future__ import annotations

import math
import warnings
from decimal import Decimal, localcontext

import numpy as np
import pytest

from affinityfit import MODELS, Dataset, fit, fit_global, hill, ic50, langmuir, michaelis, tight_binding
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
    assert set(MODELS) == {"langmuir", "hill", "michaelis", "ic50", "tight_binding"}
    for name, model in MODELS.items():
        assert isinstance(model, Model)
        assert model.name == name


def test_models_declare_consistent_roles():
    """The diagnostics are written against the role names, so every role must appear in `params`."""
    for model in MODELS.values():
        assert model.location in model.params
        assert model.amplitude in model.params
        assert model.baseline is None or model.baseline in model.params
        assert model.receptor is None or model.receptor in model.params
        assert model.exponent is None or model.exponent in model.params
        # Calling an exponent cooperative without having one would leave the claim attached to nothing.
        assert not model.cooperative or model.exponent is not None
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
    assert "hill_n_includes_one" in {diagnostic.code for diagnostic in res.warnings}


def test_hill_warns_when_n_is_significantly_below_one():
    conc, clean = hill_data(kd=10.0, n=0.4, points=20)
    # Real scatter, because a verdict of "significant" needs a spread to be significant against.
    signal = clean + np.random.default_rng(2).normal(0.0, 0.01, conc.size)
    res = fit(conc, signal, model=hill)
    assert not res.intervals["n"].zero_width
    assert "hill_n_below_one" in {diagnostic.code for diagnostic in res.warnings}


def test_no_hill_warning_for_langmuir_model():
    conc, signal = hill_data(n=1.0)
    res = fit(conc, signal, model=langmuir)
    hill_codes = {"hill_n_undetermined", "hill_n_includes_one", "hill_n_below_one", "hill_n_above_one"}
    assert not hill_codes & {diagnostic.code for diagnostic in res.warnings}


def test_hill_caveats_a_coefficient_significantly_above_one():
    """The direction people set out to claim does not pass without the alternatives being named.

    A steep curve reads as positive cooperativity, but ligand depletion, self-association
    and a reading taken before equilibrium all produce the same shape. The claim is
    caveated rather than contradicted, so a genuine result still comes through.
    """
    conc, clean = hill_data(kd=10.0, n=2.5, points=20)
    # Real scatter, because a verdict of "significant" needs a spread to be significant against.
    signal = clean + np.random.default_rng(3).normal(0.0, 0.01, conc.size)
    res = fit(conc, signal, model=hill, fixed={"baseline": 0.0})
    assert not res.intervals["n"].zero_width
    assert not res.intervals["n"].contains(1.0)
    above = [diagnostic for diagnostic in res.warnings if diagnostic.code == "hill_n_above_one"]
    assert len(above) == 1, res.warnings


def test_an_exponent_with_no_scatter_behind_it_is_not_given_a_direction():
    """Data the model fits exactly leaves no spread, and no direction can be read off that.

    The interval collapses onto the estimate, so whether it brackets 1 is decided by the
    last bit of rounding: the same noiseless curve came out just under 1 on one machine
    and just over it on another, which would have it suggest negative cooperativity in
    one place and positive in the other. Absence of scatter is absence of information,
    not certainty, so the verdict has to be that it cannot be judged.
    """
    conc, signal = hill_data(n=1.0, points=16)
    res = fit(conc, signal, model=hill, fixed={"baseline": 0.0})
    interval = res.intervals["n"]
    assert interval.zero_width

    codes = {diagnostic.code for diagnostic in res.warnings}
    assert "hill_n_undetermined" in codes
    assert not {"hill_n_below_one", "hill_n_above_one"} & codes


def test_the_coefficient_checks_are_symmetric_about_one():
    """Both directions carry an interpretation. Only one of them used to."""
    conc = np.concatenate([[0.0], np.logspace(-1, 2, 20)])
    verdicts = {}
    for n_true in (0.4, 2.5):
        signal = hill(conc, 10.0, 1.0, 0.0, n_true) + np.random.default_rng(4).normal(0.0, 0.01, conc.size)
        res = fit(conc, signal, model=hill, fixed={"baseline": 0.0})
        assert not res.intervals["n"].contains(1.0), n_true
        verdicts[n_true] = {diagnostic.code for diagnostic in res.warnings}
    assert "hill_n_below_one" in verdicts[0.4]
    assert "hill_n_above_one" in verdicts[2.5]


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


def test_michaelis_diagnostics_are_machine_readable():
    substrate = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    res = fit(substrate, michaelis(substrate, 1.6, 0.014, 0.0), model=michaelis)
    codes = {diagnostic.code for diagnostic in res.warnings}
    assert {"weakly_saturated", "no_low_conc"} <= codes


# ------------------------------------------------ IC50 (dose-response) model


def ic50_data(half=50.0, span=-100.0, top=100.0, hillslope=1.4, points=16):
    conc = np.logspace(np.log10(half / 50), np.log10(half * 50), points)
    return conc, ic50(conc, half, span, top, hillslope)


def test_ic50_shares_the_algebra_of_hill():
    conc = np.logspace(0, 4, 12)
    np.testing.assert_allclose(
        ic50(conc, 50.0, -100.0, 100.0, 1.4),
        hill(conc, 50.0, -100.0, 100.0, 1.4),
    )


def test_ic50_recovers_known_values_from_an_inhibition_curve():
    conc, response = ic50_data(half=50.0, hillslope=1.4)
    # The response runs from the top plateau down towards zero, which is how inhibition is reported.
    assert response[0] > 99.0
    assert response[-1] < 1.0
    res = fit(conc, response, model=ic50, unit="nM")
    assert res.params["ic50"] == pytest.approx(50.0, rel=1e-6)
    assert res.params["hillslope"] == pytest.approx(1.4, rel=1e-6)


def test_ic50_report_uses_the_pharmacology_labels():
    conc, response = ic50_data()
    text = fit(conc, response, model=ic50, unit="nM").report()
    assert "IC50" in text
    assert "Hill slope" in text
    assert "Kd" not in text


def test_ic50_keeps_cooperativity_advice_out_of_the_report():
    """Declaring the slope without calling it cooperative is what keeps those checks with `hill`.

    A dose-response slope near 1 is the ordinary case rather than a finding, and the
    advice to compare against `langmuir` is about a model nobody reaches for here. The
    exclusion is declared rather than a side effect of what the parameter is called, and
    the slope is still declared so that a correction assuming a slope of 1 can find it.
    """
    assert ic50.exponent == "hillslope"
    assert ic50.cooperative is False
    assert hill.exponent == "n"
    assert hill.cooperative is True
    conc, response = ic50_data(hillslope=1.0)
    noisy = response + np.random.default_rng(5).normal(0.0, 1.0, conc.size)
    res = fit(conc, noisy, model=ic50)
    hill_codes = {"hill_n_undetermined", "hill_n_includes_one", "hill_n_below_one", "hill_n_above_one"}
    assert not hill_codes & {diagnostic.code for diagnostic in res.warnings}


def test_ic50_also_describes_an_activation_curve():
    """A positive amplitude turns the same model into an EC50 measurement."""
    conc = np.logspace(0, 4, 16)
    response = ic50(conc, 50.0, 100.0, 0.0, 1.0)
    # Rising from the bottom plateau towards the top, the mirror image of the inhibition curve.
    assert response[0] < 5.0
    assert response[-1] > 95.0
    assert np.all(np.diff(response) > 0)
    res = fit(conc, response, model=ic50)
    assert res.params["ic50"] == pytest.approx(50.0, rel=1e-6)
    assert res.params["bmax"] > 0


# ------------------------------------------------------- Tight binding model


def _exact_bound_fraction(conc: float, kd: float, rt: float) -> float:
    """Fraction of receptor bound, evaluated at 60 digits to serve as a reference."""
    with localcontext() as ctx:
        ctx.prec = 60
        b = Decimal(rt) + Decimal(conc) + Decimal(kd)
        disc = b * b - 4 * Decimal(rt) * Decimal(conc)
        return float((b - disc.sqrt()) / (2 * Decimal(rt)))


def _naive_bound_fraction(conc: float, kd: float, rt: float) -> float:
    """The textbook root, kept here only so the test can show that it fails."""
    b = rt + conc + kd
    return (b - math.sqrt(b * b - 4.0 * rt * conc)) / (2.0 * rt)


def test_tight_binding_reduces_to_langmuir_without_receptor():
    """With no receptor there is nothing to deplete, and the quadratic collapses to the hyperbola."""
    conc = np.array([0.0, 0.1, 1.0, 10.0, 100.0, 1e6])
    np.testing.assert_allclose(
        tight_binding(conc, 10.0, 1.0, 0.02, 0.0),
        langmuir(conc, 10.0, 1.0, 0.02),
        rtol=1e-12,
    )


def test_tight_binding_limits_are_exact():
    assert tight_binding(np.array([0.0]), 1.0, 1.0, 0.25, 5.0)[0] == pytest.approx(0.25)
    assert tight_binding(np.array([1e12]), 1.0, 1.0, 0.0, 5.0)[0] == pytest.approx(1.0)


def test_tight_binding_keeps_precision_where_the_naive_root_collapses():
    """The conjugate form stays exact where `b - sqrt(b^2 - 4 Rt Lt)` loses its digits.

    At low concentration those two terms agree to nearly every digit they carry, so the
    direct difference returns mostly rounding noise. The test pins the improvement
    rather than only the result: it also checks that the textbook expression really does
    fail here, which is what makes a regression back to it visible.
    """
    kd = rt = 1.0
    for conc in (1e-8, 1e-12, 1e-16):
        exact = _exact_bound_fraction(conc, kd, rt)
        stable = float(tight_binding(np.array([conc]), kd, 1.0, 0.0, rt)[0])
        naive = _naive_bound_fraction(conc, kd, rt)
        assert abs(stable - exact) / exact <= 1e-14, (conc, stable, exact)
        assert abs(naive - exact) / exact > 1e-9, (conc, naive, exact)


def test_tight_binding_is_finite_across_the_parameter_domain():
    conc = np.array([0.0, 1e-18, 1e-12, 1e-6, 1.0, 1e6, 1e12])
    for exponent in range(-30, 7):
        for rt in (0.0, 1e-12, 1.0, 1e6, 1e12):
            for baseline in (0.0, -1.0, 2.5):
                out = tight_binding(conc, 10.0**exponent, 1.0, baseline, rt)
                assert np.all(np.isfinite(out)), (exponent, rt, baseline)


def test_tight_binding_fraction_stays_within_zero_and_one():
    """A fraction of the receptor cannot be negative or exceed all of it, at any parameters.

    The upper end is allowed a few epsilons of slack. The bound holds exactly in
    arithmetic, so an overshoot is rounding in the last place, and it is left there
    rather than clamped: a clamp would flatten the derivative at saturation, which
    costs the optimiser more than the last bit is worth.
    """
    conc = np.array([0.0, 1e-18, 1e-12, 1e-6, 1.0, 1e6, 1e12])
    slack = 4 * np.finfo(float).eps
    for exponent in range(-30, 7):
        for rt in (0.0, 1e-12, 1.0, 1e6, 1e12):
            # With bmax = 1 and baseline = 0 the model returns the bound fraction itself.
            fraction = tight_binding(conc, 10.0**exponent, 1.0, 0.0, rt)
            assert np.all(fraction >= 0.0), (exponent, rt, fraction)
            assert np.all(fraction <= 1.0 + slack), (exponent, rt, fraction)


def test_tight_binding_emits_no_runtime_warning_at_the_extremes():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        tight_binding(np.array([0.0, 1e-20, 1.0, 1e20]), 1e-30, 1.0, 0.0, 0.0)
        tight_binding(np.array([0.0, 1e-20, 1.0, 1e20]), 1e30, 1.0, 0.0, 1e20)


def test_tight_binding_recovers_a_kd_that_langmuir_overestimates():
    """With the receptor at five times Kd, the hyperbola reports the wrong constant.

    Depletion moves the apparent midpoint to roughly Kd + Rt/2, so `langmuir` reads
    around 3.5 where the truth is 1.0. Solving the quadratic recovers it.
    """
    kd, rt = 1.0, 5.0
    conc = np.concatenate([[0.0], np.logspace(-2, 2, 14)])
    signal = tight_binding(conc, kd, 1.0, 0.0, rt)

    quadratic = fit(conc, signal, model=tight_binding, fixed={"rt": rt, "baseline": 0.0})
    hyperbola = fit(conc, signal, model=langmuir, fixed={"baseline": 0.0})

    assert quadratic.params["kd"] == pytest.approx(kd, rel=1e-6)
    assert hyperbola.params["kd"] > 3 * kd


def test_tight_binding_estimates_the_active_receptor_concentration():
    """Leaving rt free measures how much of the receptor is actually binding.

    Depletion changes the shape of the curve and not only its midpoint, which is what
    keeps rt and Kd from simply trading off against each other.
    """
    kd, rt = 1.0, 5.0
    conc = np.concatenate([[0.0], np.logspace(-2, 2, 14)])
    signal = tight_binding(conc, kd, 1.0, 0.0, rt)
    res = fit(conc, signal, model=tight_binding, fixed={"baseline": 0.0})
    assert res.params["rt"] == pytest.approx(rt, rel=1e-4)
    assert res.params["kd"] == pytest.approx(kd, rel=1e-4)


def test_tight_binding_still_separates_rt_from_kd_under_noise():
    """The separation is not an artefact of noiseless data: the intervals still cover the truth."""
    kd, rt = 1.0, 5.0
    conc = np.concatenate([[0.0], np.logspace(-2, 2, 20)])
    clean = tight_binding(conc, kd, 1.0, 0.0, rt)
    signal = clean + np.random.default_rng(11).normal(0.0, 0.01, conc.size)
    res = fit(conc, signal, model=tight_binding, fixed={"baseline": 0.0})
    assert res.intervals["kd"].contains(kd)
    assert res.intervals["rt"].contains(rt)


def test_depletion_does_not_pass_as_positive_cooperativity():
    """Pure 1:1 data under depletion fits `hill` with a significant n > 1, and must not pass quietly.

    This is the failure the diagnostics exist for. The system has no cooperativity
    whatsoever, the fit looks excellent, and the documented test for cooperativity
    (`intervals["n"].contains(1.0)`) comes back False. Following that without a warning
    puts a claim about mechanism into a paper that the data never supported.
    """
    kd, rt = 1.0, 5.0
    conc = np.logspace(-1, 2, 10)
    signal = tight_binding(conc, kd, 1.0, 0.0, rt)

    res = fit(conc, signal, model=hill, fixed={"baseline": 0.0})
    # The trap: a significant exponent above 1 out of a system that has none.
    assert res.r_squared > 0.99
    assert not res.intervals["n"].contains(1.0)
    assert res.params["n"] > 1.3
    # What has to be said about it.
    assert "hill_n_above_one" in {diagnostic.code for diagnostic in res.warnings}


def test_depletion_with_a_cooperativity_model_reports_the_exponent_as_unusable():
    """Given the receptor concentration, the verdict covers the exponent as well as Kd.

    Reporting only the inflated Kd would leave the cooperativity reading standing, and
    naming `tight_binding` alone is a dead end because it has no exponent, so the advice
    has to send the measurement back to a dilute receptor.
    """
    kd, rt = 1.0, 5.0
    conc = np.logspace(-1, 2, 10)
    signal = tight_binding(conc, kd, 1.0, 0.0, rt)
    res = fit(conc, signal, model=hill, receptor_conc=rt, fixed={"baseline": 0.0})

    depletion = [diagnostic for diagnostic in res.warnings if diagnostic.code == "ligand_depletion"]
    assert len(depletion) == 1, res.warnings


def test_an_unchecked_receptor_concentration_is_stated_where_it_bites():
    """A significant exponent is retained but depletion is reported only when it can be checked."""
    kd, rt = 1.0, 5.0
    conc = np.logspace(-1, 2, 10)
    signal = tight_binding(conc, kd, 1.0, 0.0, rt)

    unchecked = fit(conc, signal, model=hill, fixed={"baseline": 0.0})
    assert "hill_n_above_one" in {diagnostic.code for diagnostic in unchecked.warnings}

    # Told that the receptor is dilute, the fit retains the Hill finding without a depletion warning.
    dilute = fit(conc, signal, model=hill, receptor_conc=0.01, fixed={"baseline": 0.0})
    dilute_codes = {diagnostic.code for diagnostic in dilute.warnings}
    assert "hill_n_above_one" in dilute_codes
    assert "ligand_depletion" not in dilute_codes

    # A fit with nothing to explain stays quiet either way.
    clean_conc, clean_signal = hill_data(n=1.0, points=16)
    scattered = clean_signal + np.random.default_rng(7).normal(0.0, 0.01, clean_conc.size)
    clean = fit(clean_conc, scattered, model=hill, fixed={"baseline": 0.0})
    assert clean.intervals["n"].contains(1.0)
    assert "hill_n_above_one" not in {diagnostic.code for diagnostic in clean.warnings}


def test_the_shape_warning_does_not_send_hill_back_to_hill():
    """Structured residual-shape diagnostics are emitted once per affected fit."""
    kd, rt = 1.0, 5.0
    conc = np.concatenate([[0.0], np.logspace(-2, 2, 14)])
    signal = tight_binding(conc, kd, 1.0, 0.0, rt)

    def shape_diagnostics(model):
        return [
            diagnostic
            for diagnostic in fit(conc, signal, model=model, fixed={"baseline": 0.0}).warnings
            if diagnostic.code == "residual_structure"
        ]

    assert len(shape_diagnostics(hill)) == 1
    assert len(shape_diagnostics(langmuir)) == 1


def test_tight_binding_suppresses_the_advice_to_use_itself():
    """The ligand-depletion diagnostic does not fire when `tight_binding` is in use."""
    kd, rt = 1.0, 5.0
    conc = np.concatenate([[0.0], np.logspace(-2, 2, 14)])
    signal = tight_binding(conc, kd, 1.0, 0.0, rt)
    quadratic = fit(conc, signal, model=tight_binding, receptor_conc=rt, fixed={"rt": rt, "baseline": 0.0})
    hyperbola = fit(conc, signal, model=langmuir, receptor_conc=rt, fixed={"baseline": 0.0})
    assert "ligand_depletion" not in {diagnostic.code for diagnostic in quadratic.warnings}
    assert "ligand_depletion" in {diagnostic.code for diagnostic in hyperbola.warnings}


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
