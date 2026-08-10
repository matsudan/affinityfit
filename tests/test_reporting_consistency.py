"""Consistency of what gets reported.

The criteria, the advice and the output must not contradict one another.

  - AICc, the criterion to compare models on, is shown where the comparison actually happens
    (the report of a global fit, and the advice about the Hill coefficient).
  - When the fit itself is broken, it does not claim that sharing made the estimate possible.
  - Asking for too few resamples to form a percentile interval is refused with the reason given.
"""

from __future__ import annotations

import numpy as np
import pytest

from affinityfit import Dataset, fit, fit_global, hill, langmuir, michaelis
from affinityfit.uncertainty import MIN_BOOTSTRAP_SAMPLES

CONC = np.concatenate([[0.0], np.logspace(-1, 3, 12)])
SIGNAL = langmuir(CONC, 10.0, 1.0, 0.02) + np.random.default_rng(0).normal(0, 0.01, CONC.size)
NARROW = np.array([0.2, 0.32, 0.5, 0.8, 1.1, 1.6])


# ------------------------------------------------ Displaying AICc


def test_global_report_shows_aicc():
    """AICc is needed exactly where fits with and without sharing are compared."""
    text = fit_global([Dataset("d", CONC, SIGNAL)], ci="asymptotic").report()
    line = next(line for line in text.splitlines() if "AIC" in line)
    assert "AICc" in line
    assert "AIC =" in line  # plain AIC is shown alongside for reference


def test_both_reports_agree_on_the_criteria():
    single = fit(CONC, SIGNAL, ci="asymptotic")
    combined = fit_global([Dataset("d", CONC, SIGNAL)], ci="asymptotic")
    assert single.aicc == pytest.approx(combined.aicc)
    for text in (single.report(), combined.report()):
        assert "AICc" in text


def test_hill_advice_points_at_aicc():
    conc = np.concatenate([[0.0], np.logspace(-1, 3, 15)])
    signal = langmuir(conc, 10.0, 1.0, 0.0) + np.random.default_rng(3).normal(0, 0.02, conc.size)
    res = fit(conc, signal, model=hill, ci="asymptotic")
    message = next(w for w in res.warnings if "1 を含みます" in w)
    assert "AICc で比較" in message
    assert "AIC で比較" not in message.replace("AICc で比較", "")


# ------------------------- Suppressing the NOTE on a broken fit


def broken_and_healthy(noise_b):
    conc_a = CONC
    rng = np.random.default_rng(0)
    a = langmuir(conc_a, 5.0, 1.0, 0.0) + rng.normal(0, 0.02, conc_a.size)
    b = langmuir(NARROW, 9.0, 1.0, 0.0) + np.random.default_rng(1).normal(0, noise_b, NARROW.size)
    return [Dataset("a", conc_a, a), Dataset("b", NARROW, b)]


def test_note_is_suppressed_when_the_fit_does_not_describe_the_data():
    """A message saying the value of Kd is meaningless is never emitted alongside one saying it was estimated."""
    res = fit_global(
        broken_and_healthy(noise_b=0.30),
        shared=["bmax"],
        fixed={"baseline": 0.0},
        ci="asymptotic",
    )
    assert any("意味はありません" in w for w in res.warnings)
    assert not any("推定できています" in n for n in res.notes)


def test_note_remains_when_the_fit_is_sound():
    res = fit_global(
        broken_and_healthy(noise_b=0.01),
        shared=["bmax"],
        fixed={"baseline": 0.0},
        ci="asymptotic",
    )
    assert not any("意味はありません" in w for w in res.warnings)
    assert any("推定できています" in n for n in res.notes)


def test_note_is_suppressed_when_the_amplitude_collapsed():
    """A fit whose amplitude collapsed does not claim that the estimate was possible either.

    `vmax` is kept non-negative by `michaelis`, so two noiseless decreasing datasets pin the shared
    amplitude to 0 rather than to a genuine estimate.
    """
    down1 = langmuir(CONC, 10.0, -0.8, 1.0)
    down2 = langmuir(NARROW, 9.0, -0.8, 1.0)
    res = fit_global(
        [Dataset("down1", CONC, down1), Dataset("down2", NARROW, down2)],
        model=michaelis,
        shared=["vmax"],
        fixed={"baseline": 1.0},
        ci="asymptotic",
    )
    for name in res.names:
        assert any("潰れています" in w for w in res.warnings_per[name]), name
        assert not any("推定できています" in n for n in res.notes_per[name]), name


def test_suppression_is_per_dataset():
    """Only the NOTE of the dataset that is broken gets dropped."""
    conc_a = CONC
    rng = np.random.default_rng(0)
    a = langmuir(conc_a, 5.0, 1.0, 0.0) + rng.normal(0, 0.02, conc_a.size)
    good = langmuir(NARROW, 9.0, 1.0, 0.0) + np.random.default_rng(1).normal(0, 0.01, NARROW.size)
    bad = langmuir(NARROW, 9.0, 1.0, 0.0) + np.random.default_rng(2).normal(0, 0.40, NARROW.size)
    res = fit_global(
        [Dataset("a", conc_a, a), Dataset("good", NARROW, good), Dataset("bad", NARROW, bad)],
        shared=["bmax"],
        fixed={"baseline": 0.0},
        ci="asymptotic",
    )
    assert any("意味はありません" in w for w in res.warnings_per["bad"])
    assert not any("推定できています" in n for n in res.notes_per["bad"])
    assert any("推定できています" in n for n in res.notes_per["good"])


# ----------------------------- Explaining a resample count that is too small


@pytest.mark.parametrize("n_boot", [1, 10, 50, MIN_BOOTSTRAP_SAMPLES - 1])
def test_too_few_resamples_is_rejected_with_a_reason(n_boot):
    with pytest.raises(ValueError, match="below the 100 resamples"):
        fit(CONC, SIGNAL, ci="bootstrap", n_boot=n_boot)


def test_rejection_message_offers_an_alternative():
    with pytest.raises(ValueError) as info:
        fit(CONC, SIGNAL, ci="bootstrap", n_boot=50)
    message = str(info.value)
    assert "n_boot=50" in message
    assert "ci='profile'" in message


def test_the_minimum_itself_is_accepted():
    res = fit(CONC, SIGNAL, ci="bootstrap", n_boot=MIN_BOOTSTRAP_SAMPLES)
    assert res.intervals["kd"].bounded


def test_the_limit_only_applies_to_bootstrap():
    """The other interval methods do not use n_boot, so the value is not rejected there."""
    for method in ("asymptotic", "profile"):
        res = fit(CONC, SIGNAL, ci=method, n_boot=5)
        assert res.intervals["kd"].bounded, method
