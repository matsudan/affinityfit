"""Tests that missing or non-finite input turns into an error naming the cause.

Input that slips past validation reaches the inside of scipy, where it becomes a message
that names no cause plus a numpy RuntimeWarning. Skipping a blank cell in a CSV changes
the number of points, which quietly changes the degrees of freedom and the diagnostics.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from affinityfit import Dataset, fit, fit_global, langmuir, load_csv

CONC = np.array([0.0, 1.0, 3.0, 10.0, 30.0, 100.0])
SIGNAL = langmuir(CONC, 10.0, 1.0, 0.0)


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


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


# ------------------------------------------------------ Input given as CSV


@pytest.mark.parametrize("text", ["nan", "NaN", "inf", "-inf", "Infinity"])
def test_csv_rejects_non_finite_signal_with_a_line_number(tmp_path, text):
    path = write(tmp_path, "bad.csv", f"conc,signal\n0,0.02\n1,0.11\n3,{text}\n10,0.51\n30,0.77\n")
    with pytest.raises(ValueError, match=r"bad\.csv:4: signal is not finite"):
        load_csv(path)


def test_csv_rejects_non_finite_concentration_with_a_line_number(tmp_path):
    path = write(tmp_path, "bad.csv", "conc,signal\n0,0.02\n1,0.11\nnan,0.25\n10,0.51\n30,0.77\n")
    with pytest.raises(ValueError, match=r"bad\.csv:4: concentration is not finite"):
        load_csv(path)


def test_csv_reports_a_missing_value_instead_of_dropping_the_row(tmp_path):
    """Silently discarding a blank cell changes the number of points, the degrees of freedom and the diagnostics.

    A blank concentration and a blank signal collapse to the same one-cell row once empty cells are
    filtered, so both are reported through the same message rather than being told apart.
    """
    path = write(tmp_path, "gap.csv", "conc,signal\n0,0.02\n1,0.11\n3,\n10,0.51\n30,0.77\n100,0.93\n")
    with pytest.raises(ValueError, match=r"gap\.csv:4: one of the two values is missing") as info:
        load_csv(path)
    assert "number of points" in str(info.value)

    path = write(tmp_path, "gap2.csv", "conc,signal\n0,0.02\n1,0.11\n,0.25\n10,0.51\n30,0.77\n")
    with pytest.raises(ValueError, match=r"gap2\.csv:4: one of the two values is missing"):
        load_csv(path)


def test_csv_still_skips_blank_lines_headers_and_comments(tmp_path):
    path = write(
        tmp_path,
        "ok.csv",
        "# measured 2026-01-05\nconcentration_nM,signal\n\n0,0.02\n1,0.11\n3,0.25\n10,0.51\n\n30,0.77\n100,0.93\n",
    )
    conc, signal = load_csv(path)
    assert len(conc) == 6
    assert conc[0] == 0.0 and signal[-1] == pytest.approx(0.93)
