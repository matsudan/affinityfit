"""Data loading, fit results, and diagnostics.

The diagnostics answer what a coefficient of determination cannot: given where the
measurements sit relative to the fitted half-saturation constant, is the estimate
identifiable at all? They are written against the roles a model declares
(`location`, `amplitude`, `baseline`) rather than literal parameter names, so one
set of checks serves every model.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
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
        statistic: The test statistic, in the units described above.
        p_value: One or two-sided p-value as described above, or None when the
            check has no null distribution to draw one from.
        alpha: The significance level this library itself warns at, given for
            reference. Does not apply to "residual_sign_test", which is reported
            whenever it occurs regardless of `p_value`. A stricter level, or a
            family-wise correction across several fits, can be applied instead by
            comparing `p_value` directly.
    """

    name: str
    statistic: float
    p_value: float | None
    alpha: float


@dataclass(frozen=True)
class Diagnostic:
    """A machine-readable finding about fit quality or experimental design.

    Attributes:
        code: Stable identifier for branching in calling programs. It is the API
            contract; do not branch on `message`.
        severity: ``"warning"`` for a problem or ``"note"`` for contextual
            information.
        message: Concise English explanation for humans. It may be refined without
            changing `code`.
    """

    code: str
    severity: Literal["warning", "note"]
    message: str

    def __post_init__(self) -> None:
        if self.severity not in ("warning", "note"):
            raise ValueError(f"Unsupported diagnostic severity {self.severity!r}; expected 'warning' or 'note'.")


_DIAGNOSTIC_MESSAGES = {
    "amplitude_collapsed": "The fitted amplitude has collapsed to nearly zero, so the fit is effectively flat.",
    "no_fit": "The model does not capture the data trend; the fitted location is not meaningful.",
    "poor_fit": (
        "The fit is poor for a saturation curve; the data may be noisy or the model may not match the mechanism."
    ),
    "residual_structure": "Residuals are systematically structured, so the model may not describe the mechanism.",
    "heteroscedastic": "Residual variance grows with the fitted value; pass pointwise standard deviations with sigma=.",
    "heteroscedasticity": (
        "Residual variance grows with the fitted value; pass pointwise standard deviations with sigma=."
    ),
    "param_at_bound": (
        "A fitted parameter is pinned to a model bound and is set by the constraint rather than the data."
    ),
    "not_saturated": "Saturation was not reached, so the location and amplitude cannot be identified separately.",
    "weakly_saturated": "The measured range weakly constrains the plateau, so the location interval may be broad.",
    "few_points": "There are few data points relative to the number of estimated parameters.",
    "no_points_near_kd": "No measurements lie near the fitted half-saturation concentration.",
    "one_point_near_kd": "Only one measurement lies near the fitted half-saturation concentration.",
    "kd_extrapolated": "All measurements are on the saturated side, so the location is determined by extrapolation.",
    "no_low_conc": "No sufficiently low-concentration measurement constrains the baseline.",
    "ligand_depletion": "Ligand depletion can bias the fitted location; use tight_binding for this experiment.",
    "hill_n_undetermined": "The Hill coefficient interval cannot support a conclusion about cooperativity.",
    "hill_n_includes_one": "The Hill coefficient interval includes one, so cooperativity cannot be claimed.",
    "hill_n_below_one": (
        "The Hill coefficient is significantly below one and may indicate negative cooperativity or heterogeneity."
    ),
    "hill_n_above_one": (
        "The Hill coefficient is significantly above one, but depletion, self-association, or pre-equilibrium "
        "readout can produce the same shape."
    ),
    "limit_undetermined": (
        "At least one confidence-interval limit is undetermined; report a one-sided limit instead of the "
        "point estimate."
    ),
    "no_degrees_of_freedom": "There are no degrees of freedom, so no confidence interval can be calculated.",
    "rank_deficient_jacobian": (
        "The Jacobian is rank-deficient, so the data cannot identify all parameter combinations."
    ),
    "bootstrap_insufficient_samples": "Too few bootstrap resamples converged to form a percentile interval.",
    "bootstrap_failures": "Some bootstrap resamples did not converge, so the interval may be too narrow.",
    "shared_amplitude_identifies_location": (
        "Sharing the amplitude makes this otherwise unsaturated dataset identifiable."
    ),
    "unshared_amplitude": (
        "This unsaturated dataset has a free amplitude; share it only when the maximum signal is justified as common."
    ),
}


def _diagnostic(code: str, severity: Literal["warning", "note"]) -> Diagnostic:
    return Diagnostic(code=code, severity=severity, message=_DIAGNOSTIC_MESSAGES[code])


@dataclass(frozen=True)
class FitResult:
    """Fitted parameters together with diagnostic messages.

    Attributes:
        model: The model that was fitted.
        params: `{parameter name: value}`.
        intervals: `{parameter name: Interval}`. Intervals may be asymmetric, and
            one side may be None when that limit could not be determined.
        r_squared: Coefficient of determination.
        n_points: Number of data points.
        fixed: Names and values of the parameters that were held constant.
        method: Which interval method produced `intervals`.
        aic: Akaike information criterion of the fit this came from. Present for
            reference; prefer `aicc`.
        aicc: Akaike criterion with the small-sample correction, which is the one to
            compare models on. Lower is better. For a fit over several datasets both
            describe the whole fit, not this dataset alone.
        unit: Name of the concentration unit, used for display only.
        diagnostics: Machine-readable fit findings. Branch on `Diagnostic.code`, not
            on the English `message`.
        statistics: Raw statistic and p-value behind the residual-shape and
            heteroscedasticity checks, for applying your own multiple-comparison
            correction across several fits. See `Statistic`.
    """

    model: Model
    params: dict[str, float]
    intervals: dict[str, Interval]
    r_squared: float
    n_points: int
    fixed: dict[str, float] = field(default_factory=dict)
    method: str = "profile"
    aic: float = float("nan")
    aicc: float = float("nan")
    unit: str = ""
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)
    statistics: tuple[Statistic, ...] = field(default_factory=tuple)

    @property
    def ci95(self) -> dict[str, float]:
        """Half-width per parameter, or infinity when a limit is undetermined.

        Convenient for roughly symmetric intervals. Use `intervals` when the
        interval may be skewed or one-sided.
        """
        return {name: iv.half_width for name, iv in self.intervals.items()}

    @property
    def values(self) -> tuple[float, ...]:
        """Parameter values in the positional order the model function expects."""
        return self.model.ordered(self.params)

    @property
    def location(self) -> float:
        """Value of the half-saturation parameter, whatever the model calls it."""
        return self.params[self.model.location]

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        """Warning-severity diagnostics. `diagnostics` is the canonical collection."""
        return tuple(diagnostic for diagnostic in self.diagnostics if diagnostic.severity == "warning")

    @property
    def notes(self) -> tuple[Diagnostic, ...]:
        """Note-severity diagnostics. `diagnostics` is the canonical collection."""
        return tuple(diagnostic for diagnostic in self.diagnostics if diagnostic.severity == "note")

    def predict(self, conc: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Fitted value at an arbitrary concentration."""
        return self.model(np.asarray(conc, dtype=float), *self.values)

    def residuals(self, conc: NDArray[np.float64], signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Observed minus fitted values."""
        return np.asarray(signal, dtype=float) - self.predict(np.asarray(conc, dtype=float))

    def curve(
        self,
        conc_min: float | None = None,
        conc_max: float | None = None,
        n: int = 300,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return a smooth fitted curve as `(x, y)`, log-spaced. No figure is returned.

        Linear spacing would collapse the low-concentration region and hide the
        curvature around the half-saturation point. When the range is omitted, the
        curve spans 1/100 to 100 times that point.

        Args:
            conc_min: Lower end of the concentration range. Must be positive.
            conc_max: Upper end of the concentration range.
            n: Number of points on the curve.

        Returns:
            A tuple `(x, y)` of concentrations and fitted values.

        Raises:
            ValueError: If `conc_min` is not positive, or `conc_max` <= `conc_min`.
        """
        centre = self.location
        lo = conc_min if conc_min is not None else centre / 100.0
        hi = conc_max if conc_max is not None else centre * 100.0
        if lo <= 0:
            raise ValueError("conc_min must be positive (the concentration axis is logarithmic).")
        if hi <= lo:
            raise ValueError("conc_max must be greater than conc_min.")
        x = np.logspace(np.log10(lo), np.log10(hi), n)
        return x, self.model(x, *self.values)

    def report(self) -> str:
        """Render the fitted parameters and diagnostics as human-readable text."""
        width = max(len(self.model.label(p)) for p in self.model.params)
        lines = [
            f"model    : {self.model.name}  ({self.model.description})",
            f"interval : {self.method}",
        ]
        for name in self.model.params:
            label = self.model.label(name).ljust(width)
            unit = self.unit if name == self.model.location else ""
            if name in self.fixed:
                suffix = f" {unit}" if unit else ""
                lines.append(f"{label} = {self.params[name]:.4g}{suffix}  (fixed)")
            else:
                lines.append(f"{label} = {self.intervals[name].format(unit)}")
        lines.append(f"{'R^2'.ljust(width)} = {self.r_squared:.4f}   (n = {self.n_points})")
        if np.isfinite(self.aicc):
            lines.append(f"{'AICc'.ljust(width)} = {self.aicc:.2f}   (AIC = {self.aic:.2f})")
        lines.append("")
        lines.extend(
            f"{diagnostic.severity.upper()} [{diagnostic.code}]: {diagnostic.message}"
            for diagnostic in self.diagnostics
        )
        if not self.diagnostics:
            lines.append("No diagnostic issues detected.")
        return "\n".join(lines)


def _positions(mask: NDArray[np.bool_], limit: int = 5) -> str:
    """Human-readable list of the offending indices, truncated."""
    indices = np.flatnonzero(mask).tolist()
    shown = ", ".join(str(i) for i in indices[:limit])
    return shown if len(indices) <= limit else f"{shown}, ... ({len(indices)} total)"


def _reject_non_finite(values: NDArray[np.float64], label: str, where: str) -> None:
    bad = ~np.isfinite(values)
    if bad.any():
        raise ValueError(f"{where}: {label} contains NaN or infinity at index {_positions(bad)}")


def load_csv(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read `(concentration, signal)` from a CSV file with two or more columns.

    The first column is taken as concentration and the second as signal. A row that
    cannot be parsed as numbers is skipped as a header or comment. Repeated rows at
    the same concentration are allowed and are used as replicates.

    A row that carries a number in one column and nothing in the other is an error
    rather than something to skip: dropping it would change the number of points, and
    with it the degrees of freedom and the diagnostics that depend on them.

    Args:
        path: Path to the CSV file.

    Returns:
        A tuple `(conc, signal)` of arrays.

    Raises:
        ValueError: If fewer than 3 rows are numeric, a row cannot be parsed, a value
            is missing, a value is NaN or infinite, or a concentration is negative.
    """
    conc: list[float] = []
    signal: list[float] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for lineno, row in enumerate(csv.reader(handle), start=1):
            cells = [cell.strip() for cell in row if cell.strip() != ""]
            if not cells:
                continue  # blank row
            if len(cells) < 2:
                if _looks_numeric(cells[0]):
                    raise ValueError(
                        f"{path}:{lineno}: one of the two values is missing: {row!r}. "
                        "Remove the row or fill in the measurement; dropping it silently "
                        "would change the number of points."
                    )
                continue  # header or comment
            try:
                x, y = float(cells[0]), float(cells[1])
            except ValueError:
                if lineno == 1 or not conc:
                    continue  # header row
                raise ValueError(f"{path}:{lineno}: row is not numeric: {row!r}") from None
            if not math.isfinite(x):
                raise ValueError(f"{path}:{lineno}: concentration is not finite: {cells[0]!r}")
            if not math.isfinite(y):
                raise ValueError(f"{path}:{lineno}: signal is not finite: {cells[1]!r}")
            conc.append(x)
            signal.append(y)

    if len(conc) < 3:
        raise ValueError(
            f"Only {len(conc)} data point(s) found. "
            "Fitting 3 parameters requires at least 3 points, and 6 or more in practice."
        )
    if min(conc) < 0:
        raise ValueError("Concentration contains negative values.")
    return np.asarray(conc, dtype=float), np.asarray(signal, dtype=float)


def _looks_numeric(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


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


def _is_decreasing(conc: NDArray[np.float64], signal: NDArray[np.float64]) -> bool:
    """Whether the signal falls as the concentration rises.

    Compares the mean of the lower half of the concentrations with that of the upper
    half, which is robust to noise on individual points.
    """
    order = np.argsort(conc)
    half = max(1, len(conc) // 2)
    return bool(signal[order[-half:]].mean() < signal[order[:half]].mean())


def _residual_structure(
    conc: NDArray[np.float64],
    signal: NDArray[np.float64],
    model: Model,
    params: dict[str, float],
    stats_out: list[Statistic] | None = None,
) -> list[tuple[str, str]]:
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
    pattern = "".join("+" if s > 0 else "-" for s in nonzero)

    if n_pos == 0 or n_neg == 0:
        # The runs count is degenerate here (always 1), so there is no runs statistic to report. What
        # can be reported is the exact two-sided probability, under independent coin-flip signs, that
        # every one of them would land the same way: 2 * 0.5**n, the sum of the two matching tails.
        p_sign = min(1.0, 2.0 * 0.5**nonzero.size)
        if stats_out is not None:
            stats_out.append(
                Statistic(name="residual_sign_test", statistic=float(nonzero.size), p_value=p_sign, alpha=0.05)
            )
        return [
            (
                "residual_structure",
                f"残差 {nonzero.size} 点すべてが同じ符号です。フィッティングした曲線がデータ全体から"
                "一方向にずれており、モデルが機構に合っていません。",
            )
        ]

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

    # Pointing at `hill` is only useful to someone not already fitting an exponent. Suggesting it to a
    # model that has one sends them back to what they are doing, and under ligand depletion, which is
    # one cause of this pattern, it points away from the actual explanation.
    suggestion = (
        "別の機構（リガンド枯渇なら tight_binding）を検討してください。"
        if model.exponent is not None
        else "協同性（hill）や別の機構を検討してください。"
    )
    return [
        (
            "residual_structure",
            f"残差が系統的に偏っています（符号の連 {runs} / 期待 {mean_runs:.1f}, z = {z:.2f}、"
            f"隣接残差の自己相関 = {autocorr:.2f}）。符号の並び: {pattern}。"
            "決定係数が高くてもモデルの形が機構に合っていません。" + suggestion,
        )
    ]


def _heteroscedastic(
    conc: NDArray[np.float64],
    signal: NDArray[np.float64],
    model: Model,
    params: dict[str, float],
    stats_out: list[Statistic] | None = None,
) -> list[tuple[str, str]]:
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
    return [
        (
            "heteroscedastic",
            f"残差の大きさがフィッティング値とともに増えています（順位相関 = {rho:.2f}）。"
            "誤差の大きさが点ごとに違うため、全点を等価値に扱うフィッティングでは精度が落ち、"
            "信頼区間が狭く出ることがあります。蛍光・発光・吸光のように信号に比例した"
            "誤差を持つ系では、sigma= に点ごとの標準偏差を渡してください。",
        )
    ]


def _diagnose_coded(
    conc: NDArray[np.float64],
    signal: NDArray[np.float64],
    model: Model,
    params: dict[str, float],
    intervals: dict[str, Interval] | None = None,
    receptor_conc: float | None = None,
    r_squared: float | None = None,
    fixed_names: tuple[str, ...] = (),
    weighted: bool = False,
    stats_out: list[Statistic] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return diagnostics as `(code, message)` pairs.

    The code lets callers filter messages. For example, when the amplitude is shared
    in a global fit, the remark that the half-saturation constant and the amplitude
    cannot be separated no longer applies and is therefore suppressed.

    Messages are written in Japanese because they are advice addressed to the
    person interpreting the fit, not part of the API surface.

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
        r_squared: Coefficient of determination, used to detect a model that does
            not describe the data at all.
        fixed_names: Parameters that were held constant, exempt from the
            stuck-at-a-bound check.
        weighted: Whether per-point sigma was supplied. Suppresses the
            heteroscedasticity check, which only asks whether weights are needed.
        stats_out: When given, the statistics behind the residual-shape and
            heteroscedasticity checks are appended to it, whether or not they end
            up warranting a message. See `Statistic`.
    """
    loc = float(params[model.location])
    loc_name = model.label(model.location)
    amp_name = model.label(model.amplitude)

    msgs: list[tuple[str, str]] = []
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
    decreasing = _is_decreasing(conc, signal)

    if spread > 0 and abs(amplitude) <= 0.01 * spread:
        detail = (
            f"データは濃度とともに減少していますが、{model.name} モデルの {amp_name} は"
            "非負に制限されています。減少する観測量（蛍光クエンチ、強度の減少）には "
            "langmuir または hill を使ってください。"
            if decreasing and model.lower(model.amplitude) >= 0.0
            else f"モデルがデータの形を表現できていない可能性があります（データの変動幅 {spread:.3g}）。"
        )
        msgs.append(
            (
                "amplitude_collapsed",
                f"{amp_name} = {amplitude:.3g} がほぼ 0 に潰れています。フィッティングは実質的に水平線です。{detail}",
            )
        )

    if r_squared is not None and np.isfinite(r_squared):
        if r_squared < 0.5:
            msgs.append(
                (
                    "no_fit",
                    f"決定係数 R^2 = {r_squared:.3g} が低く、モデルがデータの傾向を捉えていません。"
                    f"{loc_name} の値に意味はありません。モデルの選択（データの増減の向き、"
                    "協同性、別の機構）を確認してください。",
                )
            )
        elif r_squared < 0.9:
            msgs.append(
                (
                    "poor_fit",
                    f"決定係数 R^2 = {r_squared:.3g} は飽和曲線としては低めです。"
                    "ノイズが大きいか、モデルが機構に合っていない可能性があります。",
                )
            )

    # --- Whether the residuals have systematic structure. Even with a high coefficient of determination, a
    # biased sign pattern means the shape of the model does not match the mechanism.
    msgs.extend(_residual_structure(conc, signal, model, params, stats_out))

    # --- If the size of the error differs from point to point, say that weights should be supplied.
    if not weighted:
        msgs.extend(_heteroscedastic(conc, signal, model, params, stats_out))

    # --- A value stuck at a bound is a product of the constraint, not an estimate, so it cannot be reported.
    already = {code for code, _ in msgs}
    for name in model.params:
        if name in (fixed_names or ()):
            continue
        if name == model.amplitude and "amplitude_collapsed" in already:
            continue
        value = float(params[name])
        for bound, side in ((model.lower(name), "下限"), (model.upper(name), "上限")):
            if not np.isfinite(bound) or not _at_bound(value, bound, model.is_log_scale(name)):
                continue
            msgs.append(
                (
                    "param_at_bound",
                    f"{model.label(name)} = {value:.4g} が許容範囲の{side}"
                    f"（{bound:g}）に張り付いています。この値は推定結果ではなく制約の"
                    "産物なので、そのまま報告できません。モデルの選択か測定範囲を"
                    "見直してください。",
                )
            )
            break

    if cmax < 3 * loc:
        msgs.append(
            (
                "not_saturated",
                f"最高濃度 {cmax:.3g} が {loc_name}={loc:.3g} の 3 倍未満です。飽和に達しておらず "
                f"{loc_name} と {amp_name} を分離できません（結論は「{loc_name} > {cmax:.3g}」に留めるべきです）。",
            )
        )
    elif cmax < 10 * loc:
        msgs.append(
            (
                "weakly_saturated",
                f"最高濃度 {cmax:.3g} が {loc_name}={loc:.3g} の 10 倍未満です。{amp_name} の推定が "
                f"不安定で、{loc_name} の信頼区間も広がりがちです。",
            )
        )

    if n_points < 2 * n_estimated:
        msgs.append(
            (
                "few_points",
                f"データ点が {n_points} 点のみです（推定パラメータは {n_estimated} 個）。"
                "信頼区間は参考値として扱ってください。",
            )
        )

    near = int(np.count_nonzero((conc > loc / 3) & (conc < loc * 3)))
    if near == 0:
        msgs.append(
            (
                "no_points_near_kd",
                f"{loc_name} 近傍（{loc / 3:.3g} 〜 {loc * 3:.3g}）に測定点がありません。"
                f"この範囲に点を追加すると {loc_name} の精度が最も改善します。",
            )
        )
    elif near == 1:
        msgs.append(
            (
                "one_point_near_kd",
                f"{loc_name} 近傍（{loc / 3:.3g} 〜 {loc * 3:.3g}）の測定点が 1 点だけです。"
                "曲線の変曲点が 1 点に依存しています。",
            )
        )

    if cmin > loc:
        msgs.append(
            (
                "kd_extrapolated",
                f"最低濃度 {cmin:.3g} が既に {loc_name}={loc:.3g} を上回っており、全点が飽和側に"
                f"あります。{loc_name} は測定範囲より下への外挿で決まっているため、有効数字を"
                f"増やして報告できません。{loc_name} 以下の濃度点を追加してください。",
            )
        )

    if model.baseline is not None and not np.any(conc <= loc / 10):
        msgs.append(
            (
                "no_low_conc",
                f"{loc_name} の 1/10（{loc / 10:.3g}）以下の低濃度点がありません。"
                f"baseline が曲線と一緒に推定されるため、{amp_name} がずれる可能性があります。",
            )
        )

    # A model that declares a receptor role already solves the depletion, so recommending one would be
    # pointing at the model in use.
    if model.receptor is None and receptor_conc is not None and receptor_conc > loc / 10:
        # Depletion steepens the curve as well as shifting it, so a model that reads an exponent off
        # that steepness reports cooperativity that is not there. No model solves depletion and
        # cooperativity together, so the honest advice is to remove the depletion from the experiment
        # rather than to switch model, and saying "use tight_binding" alone would leave a dead end.
        if model.exponent is None:
            also_exponent = ""
        elif model.cooperative:
            also_exponent = (
                f"{model.label(model.exponent)} も 1 を上回る側に偏るため、この条件では協同性を"
                "判定できません（枯渇と協同性を同時に解くモデルはありません）。協同性を見るには"
                f"受容体濃度を {loc_name} の 1/10 以下にした測定が必要です。"
            )
        else:
            also_exponent = f"{model.label(model.exponent)} も 1 を上回る側に偏るため、そのまま解釈できません。"
        msgs.append(
            (
                "ligand_depletion",
                f"受容体（固定側）濃度 {receptor_conc:.3g} が {loc_name} の 1/10 を超えています。"
                f"結合によって遊離リガンドが減るため、このモデルは {loc_name} を過大評価します。"
                "tight_binding モデル（二次式）を使ってください。" + also_exponent,
            )
        )

    # Whether the exponent differs significantly from 1 is decided by whether its interval contains 1.
    # Only an exponent the model calls cooperative is interpreted this way; a dose-response slope is a
    # description of the curve, and a slope near 1 there is the ordinary case rather than a finding.
    coop = model.exponent if model.cooperative else None
    if coop is not None and intervals is not None and coop in intervals:
        n_interval = intervals[coop]
        coop_name = model.label(coop)
        # A zero-width interval means the residuals left nothing to estimate a spread from, which is an
        # absence of information rather than perfect knowledge. Reading a direction off one would turn
        # the last bit of floating-point rounding into a claim about a mechanism, so it is refused for
        # the same reason an unbounded interval is.
        if not n_interval.bounded or n_interval.zero_width:
            reason = (
                "残差にばらつきがないため信頼区間が幅を持ちません"
                if n_interval.zero_width
                else "の信頼区間の片側が決定できません"
            )
            msgs.append(
                (
                    "hill_n_undetermined",
                    f"{coop_name} = {float(params[coop]):.3g} {reason}。"
                    "協同性の有無を判定できるデータになっていません。",
                )
            )
        elif n_interval.contains(1.0):
            msgs.append(
                (
                    "hill_n_includes_one",
                    f"{coop_name} = {n_interval.format()} の信頼区間が 1 を含みます。"
                    "協同性があるとは主張できません。langmuir モデルで十分か AICc で比較してください。",
                )
            )
        elif n_interval.upper is not None and n_interval.upper < 1.0:
            msgs.append(
                (
                    "hill_n_below_one",
                    f"{coop_name} = {n_interval.format()} が有意に 1 を下回っています。負の協同性、"
                    "結合サイトの不均一性、または試料の不均一性を示唆します。",
                )
            )
        else:
            # The direction people set out to claim, and the one an artefact reproduces most easily. It
            # is caveated rather than contradicted: a steep curve does have a cooperative reading, but
            # the same shape arrives without any cooperativity at all, so the alternatives are named.
            # Depletion comes first because it is the one this library can rule out from an input.
            detail = (
                "受容体（固定側）濃度が未指定のため、枯渇によるものかを判定できていません。"
                "receptor_conc= を渡すと枯渇の有無を検査できます。"
                if receptor_conc is None
                else ""
            )
            msgs.append(
                (
                    "hill_n_above_one",
                    f"{coop_name} = {n_interval.format()} が有意に 1 を上回っています。正の協同性と"
                    "解釈できますが、協同性がなくても同じ形は生じます。リガンド枯渇（受容体濃度が "
                    f"{loc_name} に対して希薄でない）、会合・自己集合、平衡に達していない読み出しは"
                    "いずれも 1 を上回る側に偏らせます。これらを除外できるか確認してください。" + detail,
                )
            )

    # A parameter with one undetermined limit must not be reported with significant figures.
    if intervals is not None:
        undetermined = [
            model.label(name)
            for name in model.params
            if name in intervals and not intervals[name].bounded and name != "n"
        ]
        if undetermined:
            msgs.append(
                (
                    "limit_undetermined",
                    f"信頼区間の片側が決定できないパラメータがあります: {', '.join(undetermined)}。"
                    "点推定値を有効数字つきで報告せず、片側限界として報告してください。"
                    "パラメータの共有か、測定範囲の拡張が必要です。",
                )
            )

    return tuple(msgs)


def diagnose(
    conc: NDArray[np.float64],
    signal: NDArray[np.float64],
    model: Model,
    params: dict[str, float],
    intervals: dict[str, Interval] | None = None,
    receptor_conc: float | None = None,
    r_squared: float | None = None,
    fixed_names: tuple[str, ...] = (),
    weighted: bool = False,
    stats_out: list[Statistic] | None = None,
) -> tuple[Diagnostic, ...]:
    """Judge whether the estimated parameters can be trusted.

    Two families of problems are covered. The first is the health of the fit
    itself: a collapsed amplitude, a low coefficient of determination, residuals
    that are systematically arranged rather than scattered, and parameters stuck at
    the edge of their allowed range. The second is the placement of the measurements:
    saturation never reached, too few points, no points near or below the
    half-saturation constant, ligand depletion, and a Hill coefficient whose
    interval still contains 1.

    Args:
        conc: Ligand concentration.
        signal: Observed signal.
        model: The model that was fitted.
        params: Fitted parameters.
        intervals: Confidence intervals, used for the Hill coefficient check
            and to flag parameters whose limits are undetermined.
        receptor_conc: Concentration of the immobilised or fixed partner. Enables
            the ligand-depletion check when given, unless the model already solves
            the depletion itself.
        r_squared: Coefficient of determination, used to detect a model that does
            not describe the data at all.
        fixed_names: Parameters that were held constant, exempt from the
            stuck-at-a-bound check.
        weighted: Whether per-point sigma was supplied.
        stats_out: When given, the statistic and p-value behind the residual-shape
            and heteroscedasticity checks are appended to it. Pass a list to collect
            them for your own multiple-comparison correction across several fits;
            see `Statistic` and `FitResult.statistics`.

    Returns:
        A tuple of machine-readable diagnostics, empty when nothing was detected.
    """
    return tuple(
        _diagnostic(code, "warning")
        for code, _ in _diagnose_coded(
            conc, signal, model, params, intervals, receptor_conc, r_squared, fixed_names, weighted, stats_out
        )
    )
