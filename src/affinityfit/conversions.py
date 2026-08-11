"""Corrections applied to a fitted parameter after the fit.

What a fit measures is not always what gets reported. The corrections here close that
gap, and they live outside the models on purpose: each one rests on assumptions about
how the assay was run that the fit itself has no way to check.
"""

from __future__ import annotations

import math
import warnings
from typing import overload

from affinityfit.core import FitResult
from affinityfit.uncertainty import Interval, Method


def _finite(value: float | None) -> float | None:
    """The value when it is usable as a limit, otherwise None."""
    if value is None or not math.isfinite(value):
        return None
    return value


def _relative_widths(interval: Interval) -> tuple[float | None, float | None]:
    """Distance from the point estimate down to `lower` and up to `upper`, relative to the point.

    None on a side whose limit is undetermined, which is how that side stays
    undetermined all the way through the correction.
    """
    point = interval.point
    if point == 0.0 or not math.isfinite(point):
        return None, None
    lower, upper = _finite(interval.lower), _finite(interval.upper)
    down = (point - lower) / point if lower is not None else None
    up = (upper - point) / point if upper is not None else None
    return down, up


def _tracer_factor(tracer_conc: float, tracer_kd: float | Interval) -> tuple[float, float]:
    """The Cheng-Prusoff factor `1 + [T]/Kd` and the relative uncertainty it carries.

    Differentiating with respect to Kd gives

        d(1 + r) / (1 + r) = [r / (1 + r)] * dKd / Kd,   r = [T] / Kd,

    so the tracer constant's own relative error reaches Ki damped by `r / (1 + r)`. The
    damping is what keeps a tracer run far below its Kd from mattering, and what makes
    it matter almost in full once `[T]` is well above it.
    """
    point = tracer_kd.point if isinstance(tracer_kd, Interval) else tracer_kd
    if not math.isfinite(point) or point <= 0:
        raise ValueError(f"tracer_kd must be finite and positive, got {point!r}")

    ratio = tracer_conc / point
    factor = 1.0 + ratio
    if not isinstance(tracer_kd, Interval):
        return factor, 0.0

    if not tracer_kd.bounded:
        raise ValueError(
            "tracer_kd was given as an Interval with an undetermined limit, so the uncertainty "
            "it contributes cannot be worked out. Pass a bounded Interval, or pass the point "
            "estimate as a plain number to treat it as exact."
        )
    return factor, (ratio / factor) * (tracer_kd.half_width / point)


@overload
def ki_from_ic50(ic50: FitResult, tracer_conc: float, tracer_kd: float | Interval) -> Interval: ...


@overload
def ki_from_ic50(ic50: Interval, tracer_conc: float, tracer_kd: float | Interval) -> Interval: ...


@overload
def ki_from_ic50(ic50: float, tracer_conc: float, tracer_kd: Interval) -> Interval: ...


@overload
def ki_from_ic50(ic50: float, tracer_conc: float, tracer_kd: float) -> float: ...


def ki_from_ic50(
    ic50: float | Interval | FitResult,
    tracer_conc: float,
    tracer_kd: float | Interval,
) -> float | Interval:
    """Correct a displacement IC50 into the inhibition constant Ki (Cheng-Prusoff).

    A measured IC50 is not a property of the competitor alone. Raising the
    concentration of the labelled ligand that has to be displaced raises the IC50 along
    with it, so two laboratories running the same competitor at different tracer
    concentrations get different numbers. Dividing by `1 + [tracer] / Kd_tracer`
    removes that dependence, which is what makes Ki comparable between assays.

    Pass the whole `FitResult` rather than one interval out of it: only the result
    carries the slope alongside the IC50, and the standard form holds at a slope of 1.

    Give `tracer_kd` as an `Interval` when its uncertainty is known, which it usually is
    to no better than 20%. Its contribution reaches Ki damped by `r / (1 + r)` for
    `r = [T] / Kd`, so at `r = 2.5` a 20% uncertainty there outweighs a well-measured
    IC50 several times over; treating it as exact reports an interval narrower than the
    experiment supports. `tracer_conc` is taken as exact.

    Args:
        ic50: The fitted IC50, as a `FitResult`, as its `Interval`, or as a plain value.
            A `FitResult` is read through the model's `location` role, so it works for
            whichever model was fitted.
        tracer_conc: Concentration of the labelled ligand, or of the substrate in an
            enzyme assay. Must be the concentration the IC50 was measured at.
        tracer_kd: Dissociation constant of that labelled ligand, or Km of the
            substrate, in the same unit as `tracer_conc`. An `Interval` folds its
            uncertainty into the result.

    Returns:
        Ki, in the same unit as `ic50`. An `Interval` whenever either input carries one,
        otherwise a plain value. A limit that was undetermined stays undetermined, and a
        limit driven to zero or below by the combined uncertainty is reported as
        undetermined rather than as a Ki of zero, which would mean infinitely tight
        binding.

    Raises:
        ValueError: If `tracer_conc` is negative or not finite, `tracer_kd` is not
            finite and positive, or `tracer_kd` is an `Interval` with an undetermined
            limit.

    Warns:
        UserWarning: If the fitted slope's interval excludes 1, which puts the result
            outside what the standard form describes. The value is still returned,
            because the modified forms that cover a slope away from 1 do not agree with
            one another and picking one is the caller's decision, not this library's.

    Note:
        The relation assumes the competitor and the tracer exclude each other from a
        single site, and that the free tracer concentration is close to the total one.
        Neither holds for allosteric competition or for an assay where the tracer is
        itself depleted by binding, and no fit can detect the difference: the curve looks
        the same. The same expression covers competitive enzyme inhibition, with `[S]`
        and `Km` in place of the tracer; a non-competitive or uncompetitive mechanism
        needs a different correction and this one does not apply.

        Combining the two uncertainties is first order and assumes they are independent,
        which they are when the tracer constant comes from a separate experiment.
    """
    if not math.isfinite(tracer_conc) or tracer_conc < 0:
        raise ValueError(f"tracer_conc must be finite and non-negative, got {tracer_conc!r}")

    if isinstance(ic50, FitResult):
        _check_slope(ic50)
        measured: float | Interval = ic50.intervals[ic50.model.location]
    else:
        measured = ic50

    factor, factor_rel = _tracer_factor(tracer_conc, tracer_kd)

    if not isinstance(measured, Interval):
        point = measured / factor
        if factor_rel == 0.0:
            return point
        # The IC50 was handed over as exact, so the tracer constant is the only spread there is.
        return _combine(point, 0.0, 0.0, factor_rel, "asymptotic")

    return _combine(measured.point / factor, *_relative_widths(measured), factor_rel, measured.method)


def _combine(
    point: float,
    down: float | None,
    up: float | None,
    factor_rel: float,
    method: Method,
) -> Interval:
    """Add the tracer term to each side of the interval in quadrature, keeping asymmetry.

    Each side is combined on its own, so a skewed interval stays skewed and a side that
    was undetermined stays that way. A lower limit pushed to zero or below is returned
    as undetermined: Ki = 0 would read as infinitely tight binding, which is not what a
    wide interval means.
    """
    lower: float | None = None
    upper: float | None = None
    if down is not None:
        widened = math.hypot(down, factor_rel)
        lower = point * (1.0 - widened) if widened < 1.0 else None
    if up is not None:
        upper = point * (1.0 + math.hypot(up, factor_rel))
    # `method` describes where the IC50 interval came from; the tracer term is folded into it.
    return Interval(point=point, lower=lower, upper=upper, method=method)


def _check_slope(result: FitResult) -> None:
    """Warn when the fitted slope leaves the standard Cheng-Prusoff form behind."""
    exponent = result.model.exponent
    if exponent is None or exponent not in result.intervals:
        return
    slope = result.intervals[exponent]
    if not slope.bounded or slope.contains(1.0):
        return
    warnings.warn(
        f"{result.model.label(exponent)} = {slope.format()} の信頼区間が 1 を含みません。"
        "Cheng-Prusoff の標準形は 1 部位・競合・スロープ 1 を前提とするため、この条件では"
        "成立しません。返り値は標準形のまま計算しているので、Ki として報告するには"
        "べき乗を含む修正形が必要です。",
        UserWarning,
        stacklevel=3,
    )
