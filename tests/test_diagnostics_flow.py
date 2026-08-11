"""Structured diagnostics survive single/global result boundaries without string parsing."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from affinityfit import Dataset, Diagnostic, fit, fit_global, langmuir

L = np.array([0.2, 0.32, 0.5, 0.8, 1.1, 1.6])
FIXED = {"baseline": 0.0}


def two_states(noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return [
        Dataset("oxidized", L, langmuir(L, 1.1, 1.0, 0.0) + rng.normal(0, noise, L.size)),
        Dataset("reduced", L, langmuir(L, 9.0, 1.0, 0.0) + rng.normal(0, noise, L.size)),
    ]


def codes(diagnostics):
    return {diagnostic.code for diagnostic in diagnostics}


# ------------------------------------------------ carrying dataset diagnostics over


def test_result_for_carries_only_its_dataset_diagnostics():
    res = fit_global(two_states(), fixed=FIXED, ci="asymptotic")
    oxidized = res.result_for("oxidized").diagnostics
    reduced = res.result_for("reduced").diagnostics
    assert codes(oxidized) == codes(res.diagnostics_per["oxidized"])
    assert codes(reduced) == codes(res.diagnostics_per["reduced"])
    assert codes(oxidized) != codes(reduced)


def test_result_for_carries_notes_as_structured_diagnostics():
    res = fit_global(two_states(), shared=["bmax"], fixed=FIXED, ci="asymptotic")
    note = next(
        diagnostic for diagnostic in res.result_for("reduced").diagnostics if diagnostic.code.startswith("shared_")
    )
    assert note.severity == "note"
    assert note.message.isascii()


def test_global_notes_include_fit_wide_diagnostics():
    res = fit_global(two_states(), fixed=FIXED, ci="asymptotic")
    fit_note = Diagnostic("fit_context", "note", "Fit-wide context.")
    res = replace(res, fit_diagnostics=(fit_note,))
    assert res.notes == (fit_note,)
    assert res.result_for("oxidized").notes == (fit_note,)
    assert "NOTE [fit_context]: Fit-wide context." in res.report()


def test_fit_result_without_diagnostics_is_empty():
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 12)])
    signal = langmuir(conc, 10.0, 1.0, 0.02) + np.random.default_rng(0).normal(0, 0.01, conc.size)
    assert fit(conc, signal).diagnostics == ()


# ------------------------------------------ handling fit-wide diagnostics


def test_fit_level_diagnostics_reach_every_dataset():
    conc = np.array([1.0, 10.0, 100.0])
    datasets = [
        Dataset("a", conc, langmuir(conc, 10.0, 1.0, 0.0)),
        Dataset("b", conc, langmuir(conc, 40.0, 1.0, 0.0)),
    ]
    res = fit_global(datasets, ci="asymptotic")
    assert codes(res.fit_diagnostics) == {"no_degrees_of_freedom"}
    for name in res.names:
        assert "no_degrees_of_freedom" in codes(res.result_for(name).diagnostics)


def test_global_diagnostics_keep_dataset_scope():
    res = fit_global(two_states(), fixed=FIXED, ci="asymptotic")
    oxidized = codes(res.diagnostics_per["oxidized"])
    reduced = codes(res.diagnostics_per["reduced"])
    assert set(res.diagnostics_per) == {"oxidized", "reduced"}
    assert "not_saturated" in oxidized and "not_saturated" in reduced
    assert "poor_fit" not in oxidized
    assert "poor_fit" in reduced


# ------------------------------------------- consistency with fit()


def test_fit_and_result_for_agree_on_a_single_dataset():
    conc = L
    signal = langmuir(L, 9.0, 1.0, 0.0) + np.random.default_rng(1).normal(0, 0.02, L.size)
    direct = fit(conc, signal, fixed=FIXED, ci="asymptotic")
    via_global = fit_global([Dataset("data", conc, signal)], fixed=FIXED, ci="asymptotic").result_for("data")
    assert direct.diagnostics == via_global.diagnostics
    assert direct.params == via_global.params


def test_result_for_rejects_an_unknown_name():
    with pytest.raises(KeyError):
        fit_global(two_states(), fixed=FIXED, ci="asymptotic").result_for("nope")


def test_result_for_supports_curve_and_residuals():
    datasets = two_states()
    sub = fit_global(datasets, shared=["bmax"], fixed=FIXED, ci="asymptotic").result_for("oxidized")
    x, y = sub.curve()
    assert len(x) == len(y) == 300
    assert sub.residuals(datasets[0].conc, datasets[0].signal).shape == L.shape
