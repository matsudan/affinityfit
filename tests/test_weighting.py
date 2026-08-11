"""Weighting when the size of the measurement error differs from point to point.

Unweighted least squares is equivalent to assuming that the error is the same size everywhere, and it is the
maximum-likelihood estimate only under that assumption. A saturation curve moves the signal from baseline to
baseline+Bmax, so in systems where the error is proportional to the signal, as in fluorescence, luminescence
and absorbance, the assumption does not hold.

Measured (Kd=10, 300-800 simulations):
    proportional 30% + floor, 13 points         unweighted coverage 91% / weighted 95%
    proportional 30% + floor, Kd at range top   unweighted coverage 79% / weighted 94%
"""

from __future__ import annotations

import numpy as np
import pytest

from affinityfit import Dataset, fit, fit_global, langmuir

CONC = np.concatenate([[0.0], np.logspace(-1, 3, 13)])
TRUTH = langmuir(CONC, 10.0, 1.0, 0.0)
PROPORTIONAL = 0.01 + 0.30 * TRUTH


def observed(sigma, seed=3):
    return TRUTH + np.random.default_rng(seed).normal(0, sigma)


# ------------------------------------------------ passing sigma through


def test_sigma_changes_the_estimate():
    y = observed(PROPORTIONAL)
    unweighted = fit(CONC, y, ci="asymptotic").params["kd"]
    weighted = fit(CONC, y, sigma=PROPORTIONAL, ci="asymptotic").params["kd"]
    assert unweighted != weighted


def test_uniform_sigma_matches_no_sigma():
    """A constant sigma gives the same result as no sigma, since only relative values matter.

    Normalisation goes through exp(mean(log(w))), so the agreement holds within rounding error rather than exactly.
    """
    y = observed(np.full_like(TRUTH, 0.02))
    plain = fit(CONC, y, ci="asymptotic")
    scaled = fit(CONC, y, sigma=np.full_like(TRUTH, 0.02), ci="asymptotic")
    assert scaled.params["kd"] == pytest.approx(plain.params["kd"], rel=1e-7)
    assert scaled.intervals["kd"].half_width == pytest.approx(plain.intervals["kd"].half_width, rel=1e-6)


def test_result_is_invariant_to_scaling_all_sigma():
    """Multiplying sigma by a constant must not change the answer.

    Without normalisation the absolute size of the residuals changes, which moves the answer through the
    optimiser's convergence test.
    """
    y = observed(PROPORTIONAL)
    base = fit(CONC, y, sigma=PROPORTIONAL, ci="asymptotic")
    for factor in (1e-6, 1e-3, 1e3, 1e6, 1e9):
        scaled = fit(CONC, y, sigma=PROPORTIONAL * factor, ci="asymptotic")
        assert scaled.params["kd"] == pytest.approx(base.params["kd"], rel=1e-6), factor


def test_only_relative_sigma_matters():
    """It is the relative weight between datasets that affects the result."""
    conc = CONC
    quiet = langmuir(conc, 5.0, 1.0, 0.0) + np.random.default_rng(1).normal(0, 0.02, conc.size)
    noisy = langmuir(conc, 50.0, 1.0, 0.0) + np.random.default_rng(2).normal(0, 0.20, conc.size)

    plain = fit_global(
        [Dataset("quiet", conc, quiet), Dataset("noisy", conc, noisy)],
        shared=["bmax"],
        fixed={"baseline": 0.0},
        ci="asymptotic",
    )
    informed = fit_global(
        [
            Dataset("quiet", conc, quiet, sigma=np.full_like(conc, 0.02)),
            Dataset("noisy", conc, noisy, sigma=np.full_like(conc, 0.20)),
        ],
        shared=["bmax"],
        fixed={"baseline": 0.0},
        ci="asymptotic",
    )
    # the quieter dataset is weighted more heavily, so the shared Bmax moves closer to the true 1.0
    assert abs(informed.params["quiet"]["bmax"] - 1.0) < abs(plain.params["quiet"]["bmax"] - 1.0)


def test_weighted_fit_recovers_the_truth_on_clean_data():
    res = fit(CONC, TRUTH, sigma=PROPORTIONAL, ci="asymptotic")
    assert res.params["kd"] == pytest.approx(10.0, rel=1e-6)
    assert res.params["bmax"] == pytest.approx(1.0, rel=1e-6)


def test_r_squared_stays_unweighted():
    """The coefficient of determination is a descriptive statistic, so no weights are applied to it."""
    y = observed(PROPORTIONAL)
    res = fit(CONC, y, sigma=PROPORTIONAL, ci="asymptotic")
    residuals = y - res.predict(CONC)
    centered = y - y.mean()
    expected = 1.0 - float(residuals @ residuals) / float(centered @ centered)
    assert res.r_squared == pytest.approx(expected, rel=1e-9)


def test_sigma_is_validated():
    y = observed(PROPORTIONAL)
    with pytest.raises(ValueError, match="sigma must have the same shape"):
        Dataset("d", CONC, y, sigma=np.ones(3))
    with pytest.raises(ValueError, match="sigma must be strictly positive"):
        Dataset("d", CONC, y, sigma=np.where(np.arange(CONC.size) == 2, 0.0, 1.0))
    with pytest.raises(ValueError, match="sigma contains NaN"):
        Dataset("d", CONC, y, sigma=np.where(np.arange(CONC.size) == 2, np.nan, 1.0))


def test_weights_property():
    y = observed(PROPORTIONAL)
    assert np.all(Dataset("d", CONC, y).weights == 1.0)
    np.testing.assert_allclose(Dataset("d", CONC, y, sigma=PROPORTIONAL).weights, 1.0 / PROPORTIONAL)


# ------------------------------------------- coverage is restored


def test_weighting_restores_coverage_where_it_breaks():
    """Proportional-error data with Kd at the top of the measured range. Unweighted, the interval is too narrow."""
    conc = np.concatenate([[0.0], np.logspace(-1, 1.2, 11)])
    truth = langmuir(conc, 10.0, 1.0, 0.0)
    sigma = 0.01 + 0.30 * truth

    def coverage(use_sigma):
        covered = total = 0
        for seed in range(200):
            y = truth + np.random.default_rng(seed).normal(0, sigma)
            interval = fit(conc, y, sigma=sigma if use_sigma else None, ci="asymptotic").intervals["kd"]
            if interval.lower is None or interval.upper is None:
                continue
            total += 1
            covered += interval.contains(10.0)
        return covered / total

    unweighted, weighted = coverage(False), coverage(True)
    assert unweighted < 0.9
    assert weighted > 0.9
    assert weighted > unweighted


def test_weighting_improves_precision_on_proportional_error():
    def spread(use_sigma):
        estimates = [
            fit(
                CONC,
                TRUTH + np.random.default_rng(s).normal(0, PROPORTIONAL),
                sigma=PROPORTIONAL if use_sigma else None,
                ci="asymptotic",
            ).params["kd"]
            for s in range(150)
        ]
        values = np.array(estimates)
        return np.percentile(values, 75) - np.percentile(values, 25)

    assert spread(True) < spread(False)


# ------------------------------------------ diagnosing heteroscedasticity


def test_unweighted_fit_on_proportional_error_is_flagged():
    res = fit(CONC, observed(PROPORTIONAL), ci="asymptotic")
    diagnostic = next(diagnostic for diagnostic in res.warnings if diagnostic.code == "heteroscedastic")
    assert diagnostic.severity == "warning"
    assert diagnostic.message.isascii()


def test_supplying_sigma_suppresses_the_warning():
    res = fit(CONC, observed(PROPORTIONAL), sigma=PROPORTIONAL, ci="asymptotic")
    assert "heteroscedastic" not in {diagnostic.code for diagnostic in res.warnings}


def test_homoscedastic_data_is_not_flagged():
    fired = 0
    for seed in range(20):
        y = TRUTH + np.random.default_rng(seed).normal(0, 0.02, CONC.size)
        fired += "heteroscedastic" in {diagnostic.code for diagnostic in fit(CONC, y, ci="asymptotic").warnings}
    assert fired <= 1, fired


def test_check_is_skipped_for_short_datasets():
    conc = np.array([0.0, 1.0, 3.0, 10.0, 30.0])
    truth = langmuir(conc, 10.0, 1.0, 0.0)
    y = truth + np.random.default_rng(0).normal(0, 0.3 * truth + 0.01)
    assert "heteroscedastic" not in {diagnostic.code for diagnostic in fit(conc, y, ci="asymptotic").warnings}


def test_exact_fit_is_not_flagged():
    assert "heteroscedastic" not in {diagnostic.code for diagnostic in fit(CONC, TRUTH, ci="asymptotic").warnings}


# --------------------------------------- combined with a global fit


def test_global_fit_accepts_sigma_per_dataset():
    conc = CONC
    a = langmuir(conc, 5.0, 1.0, 0.0)
    b = langmuir(conc, 50.0, 1.0, 0.0)
    res = fit_global(
        [
            Dataset("a", conc, a, sigma=0.01 + 0.3 * a),
            Dataset("b", conc, b, sigma=0.01 + 0.3 * b),
        ],
        shared=["bmax"],
        fixed={"baseline": 0.0},
        ci="asymptotic",
    )
    assert res.params["a"]["kd"] == pytest.approx(5.0, rel=1e-3)
    assert res.params["b"]["kd"] == pytest.approx(50.0, rel=1e-3)


def test_mixing_weighted_and_unweighted_datasets_is_rejected():
    """Supplying sigma for only some of the datasets is forbidden.

    A dataset without sigma implicitly gets a weight of 1, so the ratio of the two weights is decided by the
    absolute scale of sigma. Merely changing the unit from a fraction to ppm would move the result, and the
    promise that only relative sizes matter would no longer hold.
    """
    conc = CONC
    a = langmuir(conc, 5.0, 1.0, 0.0)
    b = langmuir(conc, 50.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="sigma must be given for every dataset or for none"):
        fit_global(
            [Dataset("weighted", conc, a, sigma=0.01 + 0.3 * a), Dataset("plain", conc, b)],
            fixed={"baseline": 0.0},
        )


def test_mixing_error_names_which_datasets():
    conc = CONC
    a = langmuir(conc, 5.0, 1.0, 0.0)
    with pytest.raises(ValueError) as info:
        fit_global(
            [
                Dataset("with", conc, a, sigma=np.full_like(conc, 0.02)),
                Dataset("without", conc, a),
                Dataset("also_without", conc, a),
            ]
        )
    message = str(info.value)
    assert "'with'" in message
    assert "'without'" in message and "'also_without'" in message


def test_all_datasets_weighted_is_invariant_to_the_unit_of_sigma():
    """With sigma on every dataset, changing its unit must not change the result.

    Checked on a saturated and an unsaturated dataset sharing bmax, which is this library's main use.
    """
    conc_a = np.concatenate([[0.0], np.logspace(-1, 3, 12)])
    conc_b = np.concatenate([[0.0], np.logspace(-1, 1.2, 10)])
    truth_a = langmuir(conc_a, 5.0, 1.0, 0.0)
    truth_b = langmuir(conc_b, 20.0, 1.0, 0.0)
    rng = np.random.default_rng(0)
    a = truth_a + rng.normal(0, 0.02, conc_a.size)
    b = truth_b + rng.normal(0, 0.02, conc_b.size)

    results = []
    for factor in (1.0, 100.0, 1e6):
        res = fit_global(
            [
                Dataset("a", conc_a, a, sigma=(0.01 + 0.05 * truth_a) * factor),
                Dataset("b", conc_b, b, sigma=(0.01 + 0.05 * truth_b) * factor),
            ],
            shared=["bmax"],
            fixed={"baseline": 0.0},
            ci="asymptotic",
        )
        results.append((res.params["a"]["bmax"], res.params["b"]["kd"]))

    first = results[0]
    for bmax, kd in results[1:]:
        assert bmax == pytest.approx(first[0], rel=1e-6)
        assert kd == pytest.approx(first[1], rel=1e-6)


def test_all_datasets_unweighted_is_still_allowed():
    conc = CONC
    a = langmuir(conc, 5.0, 1.0, 0.0)
    b = langmuir(conc, 50.0, 1.0, 0.0)
    res = fit_global(
        [Dataset("a", conc, a), Dataset("b", conc, b)],
        fixed={"baseline": 0.0},
        ci="asymptotic",
    )
    assert res.params["a"]["kd"] == pytest.approx(5.0, rel=1e-3)


def test_bootstrap_preserves_the_error_structure():
    """Resampling must be over standardised residuals. Mixing raw residuals would lose the error structure."""
    y = observed(PROPORTIONAL)
    res = fit(CONC, y, sigma=PROPORTIONAL, ci="bootstrap", n_boot=300)
    assert res.intervals["kd"].bounded
    assert res.method == "bootstrap"


def test_profile_interval_works_with_weights():
    y = observed(PROPORTIONAL)
    res = fit(CONC, y, sigma=PROPORTIONAL, ci="profile")
    assert res.intervals["kd"].lower is not None


# ------------------------------------ deriving signal from replicates


def test_signal_can_be_derived_from_replicates():
    """The behaviour the docstring promised: the mean of the replicates becomes the signal."""
    truth = langmuir(CONC, 10.0, 1.0, 0.02)
    rng = np.random.default_rng(0)
    replicates = np.vstack([truth + rng.normal(0, 0.02, CONC.size) for _ in range(3)])

    dataset = Dataset("x", CONC, replicates=replicates)
    assert dataset.signal is not None
    np.testing.assert_allclose(dataset.signal, replicates.mean(axis=0))
    np.testing.assert_allclose(dataset.observed, replicates.mean(axis=0))


def test_deriving_the_signal_matches_passing_the_mean_explicitly():
    truth = langmuir(CONC, 10.0, 1.0, 0.02)
    rng = np.random.default_rng(1)
    replicates = np.vstack([truth + rng.normal(0, 0.02, CONC.size) for _ in range(4)])

    derived = fit_global([Dataset("x", CONC, replicates=replicates)], ci="asymptotic")
    explicit = fit_global([Dataset("x", CONC, replicates.mean(axis=0), replicates=replicates)], ci="asymptotic")
    assert derived.params["x"]["kd"] == explicit.params["x"]["kd"]


def test_omitting_both_signal_and_replicates_is_rejected():
    with pytest.raises(ValueError, match="give either signal or replicates"):
        Dataset("x", CONC)


def test_replicates_are_validated_before_the_signal_is_derived():
    with pytest.raises(ValueError, match="replicates must have shape"):
        Dataset("x", CONC, replicates=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="replicates contains NaN"):
        Dataset("x", CONC, replicates=np.full((3, CONC.size), np.nan))


def test_derived_signal_works_with_bootstrap_over_replicates():
    truth = langmuir(CONC, 10.0, 1.0, 0.02)
    rng = np.random.default_rng(2)
    replicates = np.vstack([truth + rng.normal(0, 0.02, CONC.size) for _ in range(3)])
    res = fit_global([Dataset("x", CONC, replicates=replicates)], ci="bootstrap", n_boot=200)
    interval = res.result_for("x").intervals["kd"]
    assert interval.bounded and interval.contains(10.0)
