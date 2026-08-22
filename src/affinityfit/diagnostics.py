"""Diagnostic records and coded fit-quality checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from affinityfit.models import Model
from affinityfit.uncertainty import Interval


@dataclass(frozen=True)
class Statistic:
    """A single statistical test result, for correcting across datasets yourself.

    The warnings are already thresholded at a fixed significance level, which is
    fine for an individual fit but not composable: reporting several fits in one
    paper means several tests, and deciding whether any of them is worth reporting
    then calls for a correction across them (Bonferroni, Benjamini-Hochberg, and so
    on) chosen for that specific comparison. Only the person running the study knows
    how many tests that is and what family they belong to, so the raw statistic and
    p-value are exposed here rather than baked into a threshold inside the library.

    Attributes:
        name: Which check this is.

            - "residual_runs": Wald-Wolfowitz runs test on the sign of the
              residuals. `statistic` is a z-score; `p_value` is the one-sided,
              lower-tail probability (small when there are too few runs).
            - "residual_sign_test": Used instead of the runs test when every
              residual shares one sign, which makes the runs count degenerate
              (always 1, with no variability to test). `statistic` is the number
              of residuals; `p_value` is the two-sided exact probability, under
              independent coin-flip signs, of every one of them agreeing.
            - "residual_autocorrelation": Lag-1 autocorrelation of the residuals.
              `p_value` is None; this is judged against a fixed threshold (0.3)
              rather than a null distribution.
            - "heteroscedasticity": Spearman correlation between the fitted values
              and the absolute residuals. `statistic` is the correlation
              coefficient; `p_value` is one-sided for the residuals growing with
              the fitted value specifically (near 1 when they shrink instead).
            - "model_vs_constant": F-test of the fitted model against a model with
              no free parameters beyond the mean. `statistic` is the F-statistic;
              `p_value` is the upper-tail probability (large when the model does
              not explain the data better than its own mean would).
        statistic: The test statistic, in the units described above.
        p_value: One or two-sided p-value as described above, or None when the
            check has no null distribution to draw one from.
        alpha: The significance level this library itself warns at, given for
            reference. Does not apply to "residual_sign_test", which is reported
            whenever it occurs regardless of `p_value`. Nor does it apply to
            "model_vs_constant" when the data has no variance at all (every value
            identical): the fit is then judged against that constant directly, with
            no p-value to compare. A stricter level, or a family-wise correction
            across several fits, can be applied instead by comparing `p_value`
            directly.
    """

    name: str
    statistic: float
    p_value: float | None
    alpha: float


class DiagnosticCode(StrEnum):
    """Every stable code a `Diagnostic` can carry.

    Members are plain strings (`DiagnosticCode.NOT_SATURATED == "not_saturated"`), so
    existing comparisons, set membership, and string methods on `Diagnostic.code`
    keep working. Import this enum to discover every code the library can emit,
    for example to branch exhaustively or to build a lookup table of your own:

    ```python
    from affinityfit import DiagnosticCode

    list(DiagnosticCode)  # every code this library can emit
    ```
    """

    AMPLITUDE_COLLAPSED = "amplitude_collapsed"
    NO_FIT = "no_fit"
    RESIDUAL_STRUCTURE = "residual_structure"
    HETEROSCEDASTIC = "heteroscedastic"
    PARAM_AT_BOUND = "param_at_bound"
    NOT_SATURATED = "not_saturated"
    WEAKLY_SATURATED = "weakly_saturated"
    FEW_POINTS = "few_points"
    NO_POINTS_NEAR_KD = "no_points_near_kd"
    KD_EXTRAPOLATED = "kd_extrapolated"
    NO_LOW_CONC = "no_low_conc"
    LIGAND_DEPLETION = "ligand_depletion"
    HILL_N_UNDETERMINED = "hill_n_undetermined"
    HILL_N_INCLUDES_ONE = "hill_n_includes_one"
    HILL_N_BELOW_ONE = "hill_n_below_one"
    HILL_N_ABOVE_ONE = "hill_n_above_one"
    LIMIT_UNDETERMINED = "limit_undetermined"
    NO_DEGREES_OF_FREEDOM = "no_degrees_of_freedom"
    RANK_DEFICIENT_JACOBIAN = "rank_deficient_jacobian"
    BOOTSTRAP_INSUFFICIENT_SAMPLES = "bootstrap_insufficient_samples"
    BOOTSTRAP_FAILURES = "bootstrap_failures"
    SHARED_AMPLITUDE_IDENTIFIES_LOCATION = "shared_amplitude_identifies_location"
    UNSHARED_AMPLITUDE = "unshared_amplitude"


@dataclass(frozen=True)
class Diagnostic:
    """A machine-readable finding about fit quality or experimental design.

    Attributes:
        code: Stable identifier for branching in calling programs. It is the API
            contract; do not branch on `message`. See `DiagnosticCode` for every
            value this can take.
        severity: ``"warning"`` for a problem or ``"note"`` for contextual
            information.
        message: Concise English explanation for humans. It may be refined without
            changing `code`.
    """

    code: DiagnosticCode
    severity: Literal["warning", "note"]
    message: str

    def __post_init__(self) -> None:
        if self.severity not in ("warning", "note"):
            raise ValueError(f"Unsupported diagnostic severity {self.severity!r}; expected 'warning' or 'note'.")


def _at_bound(value: float, bound: float, log_scale: bool) -> bool:
    """Whether a value sits at one of its bounds, judged on the parameter's own scale.

    An absolute tolerance would call a dissociation constant of 1e-6 equal to a lower
    bound of 1e-30, so parameters fitted on a logarithmic scale are compared by their
    exponents and the rest relative to their own magnitude.
    """
    if log_scale:
        if value <= 0 or bound <= 0:
            return value == bound
        exponent, bound_exponent = np.log10(value), np.log10(bound)
        return bool(abs(exponent - bound_exponent) <= 1e-6 * max(abs(bound_exponent), 1.0))
    scale = max(abs(bound), abs(value))
    if scale == 0.0:
        return True  # both the value and the bound are 0
    return bool(abs(value - bound) <= 1e-6 * scale)


def _residual_structure(
    conc: NDArray[np.float64],
    signal: NDArray[np.float64],
    model: Model,
    params: dict[str, float],
    stats_out: list[Statistic] | None = None,
) -> list[DiagnosticCode]:
    """Test whether the residuals are systematically arranged along the curve.

    A model with the wrong shape leaves long stretches of same-signed residuals even
    when the coefficient of determination looks respectable. Two statistics are
    combined, both computed after ordering the points by concentration.

    - Wald-Wolfowitz runs test on the signs of the residuals. Too few runs means the
      deviation is systematic. When every residual shares one sign the runs count is
      degenerate (always 1), so an exact sign test is used in its place instead.
    - Lag-1 autocorrelation of the residuals, which is the more sensitive of the two.

    Either one firing is enough, so that the sign pattern can be reported alongside a
    verdict that rests mainly on the autocorrelation.

    Returns an empty list when the test does not apply: fewer than 8 points, or
    residuals at the level of floating-point noise. When it does apply, the
    statistics behind the verdict are appended to `stats_out` if given, whether or
    not they end up warranting a message; this is what lets a caller apply its own
    significance level or multiple-comparison correction instead of the fixed one
    used here.
    """
    order = np.argsort(conc)
    fitted = model(conc[order], *model.ordered(params))
    residuals = signal[order] - fitted
    if residuals.size < 8:
        return []

    scale = max(
        float(signal.max() - signal.min()),
        float(np.abs(signal).max()),
        float(np.abs(fitted).max()),
    )
    rms = float(np.sqrt(np.mean(residuals**2)))
    if scale <= 0 or rms <= 1e-6 * scale:
        return []  # effectively an exact fit; the signs are decided by floating-point rounding.

    signs = np.sign(residuals)
    nonzero = signs[signs != 0]
    if nonzero.size < 8:
        return []
    n_pos = int(np.count_nonzero(nonzero > 0))
    n_neg = int(nonzero.size - n_pos)

    if n_pos == 0 or n_neg == 0:
        # The runs count is degenerate here (always 1), so there is no runs statistic to report. What
        # can be reported is the exact two-sided probability, under independent coin-flip signs, that
        # every one of them would land the same way: 2 * 0.5**n, the sum of the two matching tails.
        p_sign = min(1.0, 2.0 * 0.5**nonzero.size)
        if stats_out is not None:
            stats_out.append(
                Statistic(name="residual_sign_test", statistic=float(nonzero.size), p_value=p_sign, alpha=0.05)
            )
        return [DiagnosticCode.RESIDUAL_STRUCTURE]

    runs = 1 + int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))
    total = n_pos + n_neg
    mean_runs = 2.0 * n_pos * n_neg / total + 1.0
    variance = (mean_runs - 1.0) * (mean_runs - 2.0) / (total - 1.0)
    z = (runs - mean_runs) / float(np.sqrt(variance)) if variance > 0 else 0.0
    # One-sided: only a deficit of runs (z very negative) is evidence of systematic structure. The
    # existing "z >= -1.96" threshold below is exactly the one-sided 2.5% critical value of this p-value.
    p_runs = float(stats.norm.cdf(z))

    centered = residuals - residuals.mean()
    denominator = float(centered @ centered)
    autocorr = float(centered[:-1] @ centered[1:]) / denominator if denominator > 0 else 0.0

    if stats_out is not None:
        stats_out.append(Statistic(name="residual_runs", statistic=z, p_value=p_runs, alpha=0.025))
        # No null distribution is used for the threshold itself, so there is no p-value; `alpha` here is
        # a bound on `statistic` (the correlation), not a probability.
        stats_out.append(Statistic(name="residual_autocorrelation", statistic=autocorr, p_value=None, alpha=0.3))

    if z >= -1.96 and autocorr <= 0.3:
        return []

    return [DiagnosticCode.RESIDUAL_STRUCTURE]


def _no_fit(
    conc: NDArray[np.float64],
    signal: NDArray[np.float64],
    model: Model,
    params: dict[str, float],
    n_estimated: int,
    stats_out: list[Statistic] | None = None,
) -> list[DiagnosticCode]:
    """Test whether the fitted model explains the data better than its own mean would.

    A coefficient of determination cannot answer this for a nonlinear model: it
    depends on how far the measured concentrations happen to span, so the same
    mechanism can read anywhere from weak to strong depending on the range chosen,
    and a model that fits nothing at all can still land above any fixed cutoff. An
    F-test against a model with only an intercept (its own mean) asks the well-posed
    question instead: does the fitted shape explain the scatter significantly better
    than no shape at all, at the 1% level chosen to match `_heteroscedastic`.

    The statistic behind the verdict is appended to `stats_out` if given, whether or
    not it ends up warranting a message.
    """
    dof1 = n_estimated - 1
    dof2 = conc.size - n_estimated
    if dof1 < 1 or dof2 < 1:
        return []
    fitted = model(conc, *model.ordered(params))
    ss_res = float(np.sum((signal - fitted) ** 2))
    ss_tot = float(np.sum((signal - signal.mean()) ** 2))
    if ss_tot <= 0:
        # Every measurement is identical, so the constant model already explains all of it exactly. Anything
        # the fit adds on top is noise it invented, and there is no variance left to run an F-test against.
        return [] if ss_res <= 0 else [DiagnosticCode.NO_FIT]

    f_statistic = ((ss_tot - ss_res) / dof1) / (ss_res / dof2) if ss_res > 0 else float("inf")
    p_value = float(stats.f.sf(f_statistic, dof1, dof2)) if np.isfinite(f_statistic) else 0.0
    if stats_out is not None:
        stats_out.append(Statistic(name="model_vs_constant", statistic=f_statistic, p_value=p_value, alpha=0.01))
    if p_value < 0.01:
        return []
    return [DiagnosticCode.NO_FIT]


def _heteroscedastic(
    conc: NDArray[np.float64],
    signal: NDArray[np.float64],
    model: Model,
    params: dict[str, float],
    stats_out: list[Statistic] | None = None,
) -> list[DiagnosticCode]:
    """Test whether the size of the residuals grows with the fitted value.

    Unweighted least squares assumes the measurement error is the same size at every
    point. Fluorescence, luminescence and absorbance do not work that way: the error
    scales with the signal, so the points near saturation are the noisiest. Fitting
    such data without weights costs precision and, when the half-saturation constant
    sits near the top of the measured range, produces intervals that are too narrow.

    Spearman correlation between the absolute residuals and the fitted values, tested
    one-sided at the 1% level. The stricter level keeps the false-alarm rate near that
    of the other checks; at 5% it would fire on one clean fit in twenty, which would
    make the advice easy to dismiss.

    The statistic behind the verdict is appended to `stats_out` if given, whether or
    not it ends up warranting a message, so that a caller can apply its own
    significance level or multiple-comparison correction instead of the fixed one
    used here.
    """
    if conc.size < 8:
        return []
    fitted = model(conc, *model.ordered(params))
    residuals = np.abs(signal - fitted)
    if np.allclose(residuals, 0.0) or np.ptp(fitted) == 0:
        return []

    rho, p_two_sided = stats.spearmanr(fitted, residuals)
    if not np.isfinite(rho):
        return []
    # One-sided: only growing with the fitted value (rho > 0) is the failure mode weighting addresses.
    p_one_sided = p_two_sided / 2.0 if rho > 0 else 1.0 - p_two_sided / 2.0
    if stats_out is not None:
        stats_out.append(
            Statistic(name="heteroscedasticity", statistic=float(rho), p_value=float(p_one_sided), alpha=0.01)
        )
    if rho <= 0 or p_one_sided >= 0.01:
        return []
    return [DiagnosticCode.HETEROSCEDASTIC]


def _diagnose_coded(
    conc: NDArray[np.float64],
    signal: NDArray[np.float64],
    model: Model,
    params: dict[str, float],
    intervals: dict[str, Interval] | None = None,
    receptor_conc: float | None = None,
    fixed_names: tuple[str, ...] = (),
    weighted: bool = False,
    stats_out: list[Statistic] | None = None,
) -> tuple[DiagnosticCode, ...]:
    """Return the diagnostic codes that apply.

    Callers filter on these codes. For example, when the amplitude is shared in a
    global fit, the remark that the half-saturation constant and the amplitude
    cannot be separated no longer applies and is therefore suppressed.

    Args:
        conc: Ligand concentration.
        signal: Observed signal.
        model: The model that was fitted.
        params: Fitted parameters.
        intervals: Confidence intervals, used for the Hill coefficient check
            and to flag parameters whose limits are undetermined.
        receptor_conc: Concentration of the fixed partner. Ignored when the model
            declares a `receptor` role, since such a model already solves the
            depletion this would warn about.
        fixed_names: Parameters that were held constant, exempt from the
            stuck-at-a-bound check.
        weighted: Whether per-point sigma was supplied. Suppresses the
            heteroscedasticity check, which only asks whether weights are needed.
        stats_out: When given, the statistics behind the model-vs-constant,
            residual-shape and heteroscedasticity checks are appended to it,
            whether or not they end up warranting a message. See `Statistic`.
    """
    loc = float(params[model.location])

    msgs: list[DiagnosticCode] = []
    cmax = float(conc.max())
    cmin = float(conc.min())
    n_points = len(conc)
    # Fixed parameters are not counted (the same counting as the degrees of freedom in `fit_global`).
    # A shared parameter costs less than one parameter, but is counted as a whole one to stay on the safe side.
    n_estimated = len(model.params) - len(fixed_names or ())

    # --- When the model and the data do not match at all, say so before anything else. A saturation curve
    # still fits as a horizontal line with the amplitude collapsed to 0.
    spread = float(signal.max() - signal.min())
    amplitude = float(params[model.amplitude])

    if spread > 0 and abs(amplitude) <= 0.01 * spread:
        msgs.append(DiagnosticCode.AMPLITUDE_COLLAPSED)

    msgs.extend(_no_fit(conc, signal, model, params, n_estimated, stats_out))

    # --- Whether the residuals have systematic structure. Even with a high coefficient of determination, a
    # biased sign pattern means the shape of the model does not match the mechanism.
    msgs.extend(_residual_structure(conc, signal, model, params, stats_out))

    # --- If the size of the error differs from point to point, say that weights should be supplied.
    if not weighted:
        msgs.extend(_heteroscedastic(conc, signal, model, params, stats_out))

    # --- A value stuck at a bound is a product of the constraint, not an estimate, so it cannot be reported.
    already = set(msgs)
    for name in model.params:
        if name in (fixed_names or ()):
            continue
        if name == model.amplitude and DiagnosticCode.AMPLITUDE_COLLAPSED in already:
            continue
        value = float(params[name])
        for bound in (model.lower(name), model.upper(name)):
            if not np.isfinite(bound) or not _at_bound(value, bound, model.is_log_scale(name)):
                continue
            msgs.append(DiagnosticCode.PARAM_AT_BOUND)
            break

    if cmax < 3 * loc:
        msgs.append(DiagnosticCode.NOT_SATURATED)
    elif cmax < 10 * loc:
        msgs.append(DiagnosticCode.WEAKLY_SATURATED)

    if n_points < 2 * n_estimated:
        msgs.append(DiagnosticCode.FEW_POINTS)

    near = int(np.count_nonzero((conc > loc / 3) & (conc < loc * 3)))
    if near < 2:
        msgs.append(DiagnosticCode.NO_POINTS_NEAR_KD)

    if cmin > loc:
        msgs.append(DiagnosticCode.KD_EXTRAPOLATED)

    if model.baseline is not None and not np.any(conc <= loc / 10):
        msgs.append(DiagnosticCode.NO_LOW_CONC)

    # A model that declares a receptor role already solves the depletion, so recommending one would be
    # pointing at the model in use.
    if model.receptor is None and receptor_conc is not None and receptor_conc > loc / 10:
        msgs.append(DiagnosticCode.LIGAND_DEPLETION)

    # Whether the exponent differs significantly from 1 is decided by whether its interval contains 1.
    # Only an exponent the model calls cooperative is interpreted this way; a dose-response slope is a
    # description of the curve, and a slope near 1 there is the ordinary case rather than a finding.
    coop = model.exponent if model.cooperative else None
    if coop is not None and intervals is not None and coop in intervals:
        n_interval = intervals[coop]
        # A zero-width interval means the residuals left nothing to estimate a spread from, which is an
        # absence of information rather than perfect knowledge. Reading a direction off one would turn
        # the last bit of floating-point rounding into a claim about a mechanism, so it is refused for
        # the same reason an unbounded interval is.
        if not n_interval.bounded or n_interval.zero_width:
            msgs.append(DiagnosticCode.HILL_N_UNDETERMINED)
        elif n_interval.contains(1.0):
            msgs.append(DiagnosticCode.HILL_N_INCLUDES_ONE)
        elif n_interval.upper is not None and n_interval.upper < 1.0:
            msgs.append(DiagnosticCode.HILL_N_BELOW_ONE)
        else:
            # The direction people set out to claim, and the one an artefact reproduces most easily.
            msgs.append(DiagnosticCode.HILL_N_ABOVE_ONE)

    # A parameter with one undetermined limit must not be reported with significant figures. The
    # cooperative exponent is excluded: HILL_N_UNDETERMINED already covers it above, by role rather
    # than by the literal name `hill` happens to use.
    if intervals is not None:
        undetermined = any(name in intervals and not intervals[name].bounded and name != coop for name in model.params)
        if undetermined:
            msgs.append(DiagnosticCode.LIMIT_UNDETERMINED)

    return tuple(msgs)
