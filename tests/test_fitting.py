"""Tests for the global fit (sharing and fixing parameters).

The main subject is a two-state binding assay of the kind reported for a redox-sensitive
ligand: two forms of a receptor (here "oxidized" and "reduced") are titrated with the same
ligand, sharing one amplitude, and the measured range for one form falls well short of its
Kd (0.2-1.6 mM against Kd = 9.0 mM, or 0.18 times Kd), which is what makes sharing decisive.
"""

from __future__ import annotations

import numpy as np
import pytest

from affinityfit import Dataset, fit_global, langmuir

# Measured range for a two-state titration where one form saturates and the other does not.
L_TWO_STATE = np.array([0.2, 0.32, 0.5, 0.8, 1.1, 1.6])
KD_OX, KD_RED, IMAX = 1.1, 9.0, 1.0
PROTEIN_MM = 8.4e-3  # 8.4 uM


def iqr(values):
    """Interquartile range. Used to compare spread because it is robust to outliers."""
    return np.percentile(values, 75) - np.percentile(values, 25)


def sat(conc, kd, imax):
    """Saturation curve without a baseline."""
    return langmuir(conc, kd, imax, 0.0)


def two_state_datasets(noise=0.0, seed=0, conc=L_TWO_STATE):
    rng = np.random.default_rng(seed)
    out = []
    for name, kd in (("oxidized", KD_OX), ("reduced", KD_RED)):
        y = sat(conc, kd, IMAX)
        if noise:
            y = y + rng.normal(0, noise * IMAX, conc.size)
        out.append(Dataset(name, conc, y, receptor_conc=PROTEIN_MM))
    return out


# --------------------------------------------------------------- Basic behaviour


def test_shared_parameter_has_identical_value_across_datasets():
    res = fit_global(two_state_datasets(), shared=["bmax"], fixed={"baseline": 0.0})
    assert res.params["oxidized"]["bmax"] == res.params["reduced"]["bmax"]


def test_fixed_parameter_stays_exactly_at_given_value():
    res = fit_global(two_state_datasets(), shared=["bmax"], fixed={"baseline": 0.0})
    for name in res.names:
        assert res.params[name]["baseline"] == 0.0
        assert res.intervals[name]["baseline"].half_width == 0.0
    assert res.fixed == {"baseline": 0.0}


def test_number_of_free_parameters():
    ds = two_state_datasets()
    # kd free x2 + bmax shared x1 + baseline fixed = 3
    assert fit_global(ds, shared=["bmax"], fixed={"baseline": 0.0}).n_free_params == 3
    # kd free x2 + bmax free x2 + baseline fixed = 4
    assert fit_global(ds, fixed={"baseline": 0.0}).n_free_params == 4
    # everything free = 6
    assert fit_global(ds).n_free_params == 6
    # kd shared as well = 1 + 1 = 2
    assert fit_global(ds, shared=["kd", "bmax"], fixed={"baseline": 0.0}).n_free_params == 2


def test_recovers_both_kd_values_without_noise():
    res = fit_global(two_state_datasets(), shared=["bmax"], fixed={"baseline": 0.0})
    assert res.params["oxidized"]["kd"] == pytest.approx(KD_OX, rel=1e-4)
    assert res.params["reduced"]["kd"] == pytest.approx(KD_RED, rel=1e-4)
    assert res.params["oxidized"]["bmax"] == pytest.approx(IMAX, rel=1e-4)
    assert res.r_squared > 0.9999


def test_single_dataset_with_fixed_parameter_is_allowed():
    """One dataset plus a fixed parameter: an ordinary fit with one amplitude held constant."""
    conc = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
    y = sat(conc, 1.0, 0.85)
    res = fit_global([Dataset("only", conc, y)], fixed={"bmax": 0.85, "baseline": 0.0})
    assert res.n_free_params == 1
    assert res.params["only"]["bmax"] == 0.85
    assert res.params["only"]["kd"] == pytest.approx(1.0, rel=1e-4)


# ------------------------------------------- Sharing rescues identifiability


def test_sharing_bmax_rescues_unidentifiable_dataset():
    """For the reduced state the highest concentration is only 0.18 times Kd, so on its own it is undetermined.

    Check that the spread with sharing is smaller by an order of magnitude than without it.
    """
    assert L_TWO_STATE.max() / KD_RED < 0.2  # confirm the premise

    kd_shared, kd_free = [], []
    for seed in range(60):
        ds = two_state_datasets(noise=0.05, seed=seed)
        kd_shared.append(fit_global(ds, shared=["bmax"], fixed={"baseline": 0.0}).params["reduced"]["kd"])
        kd_free.append(fit_global(ds, fixed={"baseline": 0.0}).params["reduced"]["kd"])
    kd_shared, kd_free = np.array(kd_shared), np.array(kd_free)

    assert np.median(kd_shared) == pytest.approx(KD_RED, rel=0.25)
    assert iqr(kd_shared) < iqr(kd_free) / 10
    # without sharing the estimate falls far from the true value
    assert abs(np.median(kd_free) - KD_RED) > abs(np.median(kd_shared) - KD_RED)


def test_note_explains_that_sharing_enables_the_estimate():
    res = fit_global(two_state_datasets(), shared=["bmax"], fixed={"baseline": 0.0})
    assert "shared_amplitude_identifies_location" in {diagnostic.code for diagnostic in res.notes}


def test_sharing_suppresses_the_contradictory_saturation_warning():
    res = fit_global(two_state_datasets(), shared=["bmax"], fixed={"baseline": 0.0})
    assert "not_saturated" not in {diagnostic.code for diagnostic in res.warnings}
    free = fit_global(two_state_datasets(), fixed={"baseline": 0.0})
    assert "not_saturated" in {diagnostic.code for diagnostic in free.warnings}


def test_fixed_baseline_suppresses_the_baseline_warning():
    res = fit_global(two_state_datasets(), shared=["bmax"], fixed={"baseline": 0.0})
    assert "no_low_conc" not in {diagnostic.code for diagnostic in res.warnings}
    free = fit_global(two_state_datasets(), shared=["bmax"])
    assert "no_low_conc" in {diagnostic.code for diagnostic in free.warnings}


def test_actionable_advice_survives_suppression():
    res = fit_global(two_state_datasets(), shared=["bmax"], fixed={"baseline": 0.0})
    assert "no_points_near_kd" in {diagnostic.code for diagnostic in res.warnings}


def test_warns_and_suggests_sharing_when_bmax_is_free():
    res = fit_global(two_state_datasets(), fixed={"baseline": 0.0})
    assert "unshared_amplitude" in {diagnostic.code for diagnostic in res.warnings}


def test_aicc_prefers_sharing_when_bmax_is_truly_common():
    """Checked over several noise realisations; a single seed would hide a success that was luck.

    In this setup (n = 12, k = 3 when shared and k = 4 when not) n/k is only 3 to 4, so the uncorrected
    AIC leans towards the side with more parameters. The threshold comes from the selection rate over 40 draws.
    """
    hits = 0
    for seed in range(40):
        ds = two_state_datasets(noise=0.05, seed=seed)
        shared = fit_global(ds, shared=["bmax"], fixed={"baseline": 0.0}, ci="asymptotic")
        free = fit_global(ds, fixed={"baseline": 0.0}, ci="asymptotic")
        hits += shared.aicc < free.aicc
    assert hits >= 36, hits


def test_aicc_is_stricter_than_aic_on_the_same_fit():
    """The correction term is always positive, so AICc is at least as large as AIC."""
    ds = two_state_datasets(noise=0.05, seed=0)
    results = [
        fit_global(ds, fixed={"baseline": 0.0}, ci="asymptotic"),
        fit_global(ds, shared=["bmax"], fixed={"baseline": 0.0}, ci="asymptotic"),
    ]
    for res in results:
        assert res.aicc > res.aic


def test_aicc_correction_shrinks_as_the_sample_grows():
    """AICc is only meaningfully stricter than AIC at small n/k; the gap should vanish for a large sample.

    This is checked without recomputing `_corrected_aic`'s own formula, so a bug in that formula (say, a
    wrong exponent) would not be mirrored into the expectation here.
    """
    small = two_state_datasets(noise=0.05, seed=0, conc=L_TWO_STATE)
    large_conc = np.concatenate([[0.0], np.logspace(-1, 3, 200)])
    large = two_state_datasets(noise=0.05, seed=0, conc=large_conc)

    small_res = fit_global(small, fixed={"baseline": 0.0}, ci="asymptotic")
    large_res = fit_global(large, fixed={"baseline": 0.0}, ci="asymptotic")

    small_gap = small_res.aicc - small_res.aic
    large_gap = large_res.aicc - large_res.aic
    assert small_gap > 0
    assert large_gap > 0
    assert large_gap < small_gap / 10


def test_aicc_is_infinite_when_the_sample_cannot_support_the_parameters():
    conc = np.array([1.0, 10.0, 100.0])
    res = fit_global([Dataset("d", conc, langmuir(conc, 10.0, 1.0, 0.0))], ci="asymptotic")
    assert res.n_points - res.n_free_params - 1 <= 0
    assert res.aicc == np.inf


def test_aicc_reaches_the_per_dataset_result():
    ds = two_state_datasets(noise=0.05, seed=0)
    res = fit_global(ds, shared=["bmax"], fixed={"baseline": 0.0}, ci="asymptotic")
    sub = res.result_for("reduced")
    assert sub.aicc == res.aicc
    assert sub.aic == res.aic


# --------------------------------------------------------------- API and validation


def test_result_for_returns_usable_fitresult():
    res = fit_global(two_state_datasets(noise=0.02, seed=5), shared=["bmax"], fixed={"baseline": 0.0})
    sub = res.result_for("oxidized")
    assert sub.params["kd"] == res.params["oxidized"]["kd"]
    assert sub.n_points == len(L_TWO_STATE)
    x, y = sub.curve()
    assert len(x) == len(y) == 300
    assert sub.predict(sub.location) == pytest.approx(sub.params["bmax"] / 2, rel=1e-9)


def test_report_contains_dataset_names_and_markers():
    res = fit_global(two_state_datasets(noise=0.02, seed=2), shared=["bmax"], fixed={"baseline": 0.0}, unit="mM")
    text = res.report()
    assert "oxidized" in text and "reduced" in text
    assert "fixed" in text and "shared: bmax" in text and "AIC" in text


def test_rejects_unknown_parameter_name():
    with pytest.raises(ValueError, match="Unknown parameter name"):
        fit_global(two_state_datasets(), shared=["imax"])


def test_rejects_shared_and_fixed_overlap():
    with pytest.raises(ValueError, match="both shared and fixed"):
        fit_global(two_state_datasets(), shared=["bmax"], fixed={"bmax": 1.0})


def test_rejects_all_parameters_fixed():
    with pytest.raises(ValueError, match="nothing to estimate"):
        fit_global(two_state_datasets(), fixed={"kd": 1.0, "bmax": 1.0, "baseline": 0.0})


def test_rejects_duplicate_dataset_names():
    ds = two_state_datasets()
    dup = [ds[0], Dataset("oxidized", ds[1].conc, ds[1].signal)]
    with pytest.raises(ValueError, match="Duplicate dataset names"):
        fit_global(dup)


def test_rejects_empty_dataset_list():
    with pytest.raises(ValueError, match="No datasets were given"):
        fit_global([])


def test_dataset_validates_shape_and_values():
    with pytest.raises(ValueError, match="negative values"):
        Dataset("a", np.array([-1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="data point"):
        Dataset("a", np.array([1.0]), np.array([1.0]))


def test_three_datasets_with_shared_bmax():
    conc = np.array([0.1, 0.3, 1.0, 3.0, 10.0])
    ds = [Dataset(f"cond{i}", conc, sat(conc, kd, 2.0)) for i, kd in enumerate((0.5, 2.0, 8.0))]
    res = fit_global(ds, shared=["bmax"], fixed={"baseline": 0.0})
    assert res.n_free_params == 4  # kd x3 + bmax x1
    for name, kd in zip(res.names, (0.5, 2.0, 8.0)):
        assert res.params[name]["kd"] == pytest.approx(kd, rel=1e-3)
