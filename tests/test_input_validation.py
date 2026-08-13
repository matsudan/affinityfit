"""Tests that missing or non-finite input turns into an error naming the cause.

Input that slips past validation reaches the inside of scipy, where it becomes a message
that names no cause plus a numpy RuntimeWarning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from affinityfit import Dataset, fit, fit_global, langmuir

CONC = np.array([0.0, 1.0, 3.0, 10.0, 30.0, 100.0])
SIGNAL = langmuir(CONC, 10.0, 1.0, 0.0)


# --------------------------------------------------- Input given as arrays


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_signal_names_the_column_and_index(bad):
    signal = SIGNAL.copy()
    signal[3] = bad
    with pytest.raises(ValueError, match=r"signal contains NaN or infinity at index 3"):
        fit(CONC, signal)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_concentration_names_the_column_and_index(bad):
    conc = CONC.copy()
    conc[2] = bad
    with pytest.raises(ValueError, match=r"concentration contains NaN or infinity at index 2"):
        fit(conc, SIGNAL)


def test_several_bad_positions_are_listed():
    signal = np.array([np.nan, 0.1, np.nan, 0.5, np.nan, 0.9])
    with pytest.raises(ValueError, match=r"index 0, 2, 4"):
        fit(CONC, signal)


def test_many_bad_positions_are_truncated_with_a_count():
    signal = np.full(12, np.nan)
    conc = np.linspace(0.0, 100.0, 12)
    with pytest.raises(ValueError, match=r"\.\.\. \(12 total\)"):
        fit(conc, signal)


def test_the_error_names_the_dataset():
    signal = SIGNAL.copy()
    signal[1] = np.nan
    with pytest.raises(ValueError, match=r"^reduced: signal"):
        fit_global([Dataset("reduced", CONC, signal)])


def test_no_runtime_warning_leaks_before_the_error():
    """No numpy warning is raised ahead of the validation."""
    signal = SIGNAL.copy()
    signal[3] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="NaN or infinity"):
            fit(CONC, signal)


def test_non_finite_receptor_concentration_is_rejected():
    with pytest.raises(ValueError, match="receptor_conc is not finite"):
        fit(CONC, SIGNAL, receptor_conc=float("nan"))


def test_non_finite_replicates_are_rejected():
    replicates = np.array([SIGNAL, np.where(np.arange(CONC.size) == 2, np.nan, SIGNAL)])
    with pytest.raises(ValueError, match="replicates contains NaN or infinity"):
        fit(CONC, SIGNAL, ci="bootstrap", replicates=replicates)


def test_clean_input_is_unaffected():
    res = fit(CONC, SIGNAL)
    assert res.params["kd"] == pytest.approx(10.0, rel=1e-6)
