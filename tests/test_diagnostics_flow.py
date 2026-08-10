"""Diagnostics are not lost on the way out of a fit.

Because `FitResult.report()` is built to state outright that no problems were detected, a warning dropped by
`GlobalFitResult.result_for()` turns into a false assurance.
"""

from __future__ import annotations

import numpy as np
import pytest

from bindfit import Dataset, fit, fit_global, langmuir

# Measured range that reaches only 1.6 mM, against Kd = 9.0 mM for the reduced state.
L = np.array([0.2, 0.32, 0.5, 0.8, 1.1, 1.6])
FIXED = {"baseline": 0.0}


def two_states(noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return [
        Dataset("oxidized", L, langmuir(L, 1.1, 1.0, 0.0) + rng.normal(0, noise, L.size)),
        Dataset("reduced", L, langmuir(L, 9.0, 1.0, 0.0) + rng.normal(0, noise, L.size)),
    ]


# ------------------------------------------------- carrying the warnings over


def test_result_for_carries_the_warnings_of_that_dataset():
    res = fit_global(two_states(), fixed=FIXED, ci="asymptotic")
    sub = res.result_for("reduced")
    assert sub.warnings
    assert len(sub.warnings) == len(res.warnings_per["reduced"])


def test_result_for_does_not_claim_the_fit_is_clean():
    res = fit_global(two_states(), fixed=FIXED, ci="asymptotic")
    text = res.result_for("reduced").report()
    assert "問題は検出されませんでした" not in text
    assert "WARNING:" in text


def test_result_for_does_not_leak_another_datasets_warnings():
    res = fit_global(two_states(), fixed=FIXED, ci="asymptotic")
    oxidized = res.result_for("oxidized").warnings
    reduced = res.result_for("reduced").warnings
    # no remark that was not raised for the oxidized state may creep in
    assert set(oxidized) == set(res.warnings_per["oxidized"])
    assert set(reduced) == set(res.warnings_per["reduced"])
    assert oxidized != reduced


def test_result_for_strips_the_dataset_prefix():
    res = fit_global(two_states(), fixed=FIXED, ci="asymptotic")
    for message in res.result_for("reduced").warnings:
        assert not message.startswith("[")


def test_flat_warnings_keep_the_dataset_prefix():
    res = fit_global(two_states(), fixed=FIXED, ci="asymptotic")
    assert any(w.startswith("[reduced] ") for w in res.warnings)
    assert any(w.startswith("[oxidized] ") for w in res.warnings)


# --------------------------------------------------- carrying the notes over


def test_result_for_carries_notes():
    res = fit_global(two_states(), shared=["bmax"], fixed=FIXED, ci="asymptotic")
    sub = res.result_for("reduced")
    assert sub.notes
    assert any("共有をやめると決定不能" in n for n in sub.notes)


def test_report_shows_notes_and_does_not_claim_the_fit_is_clean():
    res = fit_global(two_states(), shared=["bmax"], fixed=FIXED, ci="asymptotic")
    text = res.result_for("reduced").report()
    assert "NOTE:" in text
    assert "問題は検出されませんでした" not in text


def test_fit_result_without_any_message_still_says_so():
    """The converse of a false positive: a sound fit gets the affirmative statement."""
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 12)])
    signal = langmuir(conc, 10.0, 1.0, 0.02) + np.random.default_rng(0).normal(0, 0.01, conc.size)
    res = fit(conc, signal)
    assert res.warnings == () and res.notes == ()
    assert "問題は検出されませんでした" in res.report()


# ------------------------------ handling of warnings that concern the whole fit


def test_fit_level_warnings_reach_every_dataset():
    """A fit-wide problem such as zero degrees of freedom appears whichever dataset is pulled out."""
    conc = np.array([1.0, 10.0, 100.0])
    datasets = [
        Dataset("a", conc, langmuir(conc, 10.0, 1.0, 0.0)),
        Dataset("b", conc, langmuir(conc, 40.0, 1.0, 0.0)),
    ]
    res = fit_global(datasets, ci="asymptotic")
    assert res.fit_warnings
    for name in ("a", "b"):
        assert any("自由度" in w for w in res.result_for(name).warnings), name


def test_fit_level_warnings_appear_once_in_the_flat_view():
    conc = np.array([1.0, 10.0, 100.0])
    datasets = [
        Dataset("a", conc, langmuir(conc, 10.0, 1.0, 0.0)),
        Dataset("b", conc, langmuir(conc, 40.0, 1.0, 0.0)),
    ]
    res = fit_global(datasets, ci="asymptotic")
    assert sum("自由度" in w for w in res.warnings) == 1


# ------------------------------------------- consistency with fit()


def test_fit_and_result_for_agree_on_a_single_dataset():
    """fit() goes through result_for, so the diagnostics from the two paths agree."""
    conc, signal = L, langmuir(L, 9.0, 1.0, 0.0) + np.random.default_rng(1).normal(0, 0.02, L.size)
    direct = fit(conc, signal, fixed=FIXED, ci="asymptotic")
    viaglobal = fit_global([Dataset("data", conc, signal)], fixed=FIXED, ci="asymptotic").result_for("data")
    assert direct.warnings == viaglobal.warnings
    assert direct.notes == viaglobal.notes
    assert direct.params == viaglobal.params


def test_result_for_still_rejects_an_unknown_name():
    res = fit_global(two_states(), fixed=FIXED, ci="asymptotic")
    with pytest.raises(KeyError):
        res.result_for("nope")


def test_result_for_supports_curve_and_residuals():
    """The path shown in the README keeps working."""
    datasets = two_states()
    res = fit_global(datasets, shared=["bmax"], fixed=FIXED, ci="asymptotic")
    sub = res.result_for("oxidized")
    x, y = sub.curve()
    assert len(x) == len(y) == 300
    oxidized = datasets[0]
    assert sub.residuals(oxidized.conc, oxidized.signal).shape == L.shape
