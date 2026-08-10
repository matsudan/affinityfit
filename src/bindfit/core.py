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

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from bindfit.models import Model
from bindfit.uncertainty import Interval


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
        warnings: Problems detected in the fit.
        notes: Remarks that are informative rather than problems, such as a shared
            parameter having enabled an otherwise unidentifiable estimate.
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
    warnings: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

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

        Examples:
            >>> ax.plot(*result.curve())   # doctest: +SKIP
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
        lines.extend(f"NOTE: {n}" for n in self.notes)
        lines.extend(f"WARNING: {w}" for w in self.warnings)
        if not self.warnings and not self.notes:
            lines.append("診断チェック: 問題は検出されませんでした。")
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
) -> list[tuple[str, str]]:
    """Test whether the residuals are systematically arranged along the curve.

    A model with the wrong shape leaves long stretches of same-signed residuals even
    when the coefficient of determination looks respectable. Two statistics are
    combined, both computed after ordering the points by concentration.

    - Wald-Wolfowitz runs test on the signs of the residuals. Too few runs means the
      deviation is systematic.
    - Lag-1 autocorrelation of the residuals, which is the more sensitive of the two.

    Either one firing is enough, so that the sign pattern can be reported alongside a
    verdict that rests mainly on the autocorrelation.

    Returns an empty list when the test does not apply: fewer than 8 points, or
    residuals at the level of floating-point noise.
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
        return [
            (
                "residual_structure",
                f"残差 {nonzero.size} 点すべてが同じ符号です。当てはめた曲線がデータ全体から"
                "一方向にずれており、モデルが機構に合っていません。",
            )
        ]

    runs = 1 + int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))
    total = n_pos + n_neg
    mean_runs = 2.0 * n_pos * n_neg / total + 1.0
    variance = (mean_runs - 1.0) * (mean_runs - 2.0) / (total - 1.0)
    z = (runs - mean_runs) / float(np.sqrt(variance)) if variance > 0 else 0.0

    centered = residuals - residuals.mean()
    denominator = float(centered @ centered)
    autocorr = float(centered[:-1] @ centered[1:]) / denominator if denominator > 0 else 0.0

    if z >= -1.96 and autocorr <= 0.3:
        return []

    return [
        (
            "residual_structure",
            f"残差が系統的に偏っています（符号の連 {runs} / 期待 {mean_runs:.1f}, z = {z:.2f}、"
            f"隣接残差の自己相関 = {autocorr:.2f}）。符号の並び: {pattern}。"
            "決定係数が高くてもモデルの形が機構に合っていません。"
            "協同性（hill）や別の機構を検討してください。",
        )
    ]


def _heteroscedastic(
    conc: NDArray[np.float64],
    signal: NDArray[np.float64],
    model: Model,
    params: dict[str, float],
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
    """
    if conc.size < 8:
        return []
    fitted = model(conc, *model.ordered(params))
    residuals = np.abs(signal - fitted)
    if np.allclose(residuals, 0.0) or np.ptp(fitted) == 0:
        return []

    rho, p_two_sided = stats.spearmanr(fitted, residuals)
    if not np.isfinite(rho) or rho <= 0 or p_two_sided / 2.0 >= 0.01:
        return []
    return [
        (
            "heteroscedastic",
            f"残差の大きさが当てはめ値とともに増えています（順位相関 = {rho:.2f}）。"
            "誤差の大きさが点ごとに違うため、全点を等価値に扱う当てはめでは精度が落ち、"
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
        receptor_conc: Concentration of the fixed partner.
        r_squared: Coefficient of determination, used to detect a model that does
            not describe the data at all.
        fixed_names: Parameters that were held constant, exempt from the
            stuck-at-a-bound check.
        weighted: Whether per-point sigma was supplied. Suppresses the
            heteroscedasticity check, which only asks whether weights are needed.
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
                f"{amp_name} = {amplitude:.3g} がほぼ 0 に潰れています。当てはめは実質的に水平線です。{detail}",
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
    msgs.extend(_residual_structure(conc, signal, model, params))

    # --- If the size of the error differs from point to point, say that weights should be supplied.
    if not weighted:
        msgs.extend(_heteroscedastic(conc, signal, model, params))

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

    if receptor_conc is not None and receptor_conc > loc / 10:
        msgs.append(
            (
                "ligand_depletion",
                f"受容体（固定側）濃度 {receptor_conc:.3g} が {loc_name} の 1/10 を超えています。"
                f"結合によって遊離リガンドが減るため、このモデルは {loc_name} を過大評価します。"
                "tight-binding（二次式）モデルが必要です。",
            )
        )

    # Whether the Hill coefficient differs significantly from 1 is decided by whether its interval contains 1.
    if "n" in model.params and intervals is not None and "n" in intervals:
        n_interval = intervals["n"]
        n_hill = float(params["n"])
        if not n_interval.bounded:
            msgs.append(
                (
                    "hill_n_undetermined",
                    f"Hill 係数 n = {n_hill:.3g} の信頼区間の片側が決定できません。"
                    "協同性の有無を判定できるデータになっていません。",
                )
            )
        elif n_interval.contains(1.0):
            msgs.append(
                (
                    "hill_n_includes_one",
                    f"Hill 係数 n = {n_interval.format()} の信頼区間が 1 を含みます。"
                    "協同性があるとは主張できません。langmuir モデルで十分か AICc で比較してください。",
                )
            )
        elif n_interval.upper is not None and n_interval.upper < 1.0:
            msgs.append(
                (
                    "hill_n_below_one",
                    f"Hill 係数 n = {n_interval.format()} が有意に 1 を下回っています。負の協同性、"
                    "結合サイトの不均一性、または試料の不均一性を示唆します。",
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
) -> tuple[str, ...]:
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
            the ligand-depletion check when given.
        r_squared: Coefficient of determination, used to detect a model that does
            not describe the data at all.
        fixed_names: Parameters that were held constant, exempt from the
            stuck-at-a-bound check.
        weighted: Whether per-point sigma was supplied.

    Returns:
        A tuple of diagnostic messages, empty when nothing was detected.
    """
    return tuple(
        m
        for _, m in _diagnose_coded(
            conc, signal, model, params, intervals, receptor_conc, r_squared, fixed_names, weighted
        )
    )
