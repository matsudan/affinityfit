"""Corrections applied to a fitted parameter after the fit.

What a fit measures is not always what gets reported. The corrections here close that
gap, and they live outside the models on purpose: each one rests on assumptions about
how the assay was run that the fit itself has no way to check.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import overload

from affinityfit.core import FitResult
from affinityfit.uncertainty import Interval, Method


def _finite(value: float | None) -> float | None:
    """The value when it is usable as a limit, otherwise None."""
    if value is None or not math.isfinite(value):
        return None
    return value


def _tracer_kd_point(tracer_kd: float | Interval) -> float:
    point = tracer_kd.point if isinstance(tracer_kd, Interval) else tracer_kd
    if not math.isfinite(point) or point <= 0:
        raise ValueError(f"tracer_kd must be finite and positive, got {point!r}")
    if isinstance(tracer_kd, Interval) and not tracer_kd.bounded:
        raise ValueError(
            "tracer_kd was given as an Interval with an undetermined limit, so the uncertainty "
            "it contributes cannot be worked out. Pass a bounded Interval, or pass the point "
            "estimate as a plain number to treat it as exact."
        )
    return point


def _bound_tracer(receptor_conc: float, tracer_conc: float, tracer_kd: float) -> float:
    """Tracer bound with no competitor present, `[RL]0`, from the 1:1 equilibrium.

    The same quadratic the `tight_binding` model solves, in the form that keeps its
    precision: evaluated directly the physical root subtracts two nearly equal numbers.
    """
    total = receptor_conc + tracer_conc + tracer_kd
    if total <= 0:
        return 0.0
    u = receptor_conc / total
    v = tracer_conc / total
    return receptor_conc * 2.0 * v / (1.0 + math.sqrt(max(0.0, 1.0 - 4.0 * u * v)))


def _converter(
    tracer_conc: float,
    tracer_kd: float,
    receptor_conc: float | None,
) -> Callable[[float], float]:
    """Build the IC50 -> Ki map, exact when the receptor concentration is known.

    Without `receptor_conc` this is the Cheng-Prusoff form `IC50 / (1 + [T]/Kd)`, which
    assumes both that the free tracer concentration equals the total one and that the
    competitor bound to the receptor is a negligible part of what was added.

    With `receptor_conc` neither assumption is needed:

        Ki = (IC50 - [R]T/2) / (2[L]50/[L]0 - 1 + [L]50/Kd)

    where `[L]0` is the free tracer with no competitor and `[L]50 = [L]0 + [RL]0/2` is
    the free tracer at half displacement. This agrees with the exact Munson-Rodbard
    correction. Both are affine in IC50, so interval limits map through either directly.
    """
    if receptor_conc is None:
        factor = 1.0 + tracer_conc / tracer_kd
        return lambda ic50: ic50 / factor

    bound = _bound_tracer(receptor_conc, tracer_conc, tracer_kd)
    free = tracer_conc - bound
    if free <= 0:
        raise ValueError(
            f"receptor_conc={receptor_conc!r} leaves no free tracer at "
            f"tracer_conc={tracer_conc!r}; the correction cannot be applied. Raise the "
            "tracer concentration or lower the receptor concentration."
        )
    half = free + bound / 2.0
    denominator = 2.0 * half / free - 1.0 + half / tracer_kd
    offset = receptor_conc / 2.0
    return lambda ic50: (ic50 - offset) / denominator


def _map_limits(
    measured: float | Interval,
    convert: Callable[[float], float],
) -> tuple[float, float | None, float | None]:
    """Convert the IC50 and return its limits as relative widths of the resulting Ki.

    Both forms are affine in IC50, so each limit maps through directly. Relative widths
    are what the tracer term is later added to in quadrature, and `None` on a side keeps
    that side undetermined all the way through.

    A limit that maps to zero or below is returned as undetermined rather than as a Ki of
    zero, which would read as infinitely tight binding. The exact form subtracts
    `[R]T/2`, so this is reachable for a competitor tight enough that its IC50 approaches
    half the receptor concentration.
    """

    def positive(value: float) -> float:
        if value <= 0:
            raise ValueError(
                "The corrected Ki is not positive, so the IC50 is too close to half the "
                "receptor concentration for the correction to resolve it."
            )
        return value

    if not isinstance(measured, Interval):
        # Nothing was claimed about the spread, so the tracer term is the only source of one.
        return positive(convert(measured)), 0.0, 0.0

    point = positive(convert(measured.point))
    lower, upper = _finite(measured.lower), _finite(measured.upper)
    low = convert(lower) if lower is not None else None
    high = convert(upper) if upper is not None else None
    down = (point - low) / point if low is not None and low > 0 else None
    up = (high - point) / point if high is not None else None
    return point, down, up


def _tracer_sensitivity(
    measured_point: float,
    tracer_conc: float,
    tracer_kd: float | Interval,
    receptor_conc: float | None,
) -> float:
    """Relative half-width that the tracer constant's own error contributes to Ki.

    For the Cheng-Prusoff form this is closed form. Differentiating with respect to Kd
    gives

        d(1 + r) / (1 + r) = [r / (1 + r)] * dKd / Kd,   r = [T] / Kd,

    so the relative error of the tracer constant reaches Ki damped by `r / (1 + r)`. The
    damping keeps a tracer run far below its own Kd from mattering, and makes it matter
    almost in full once `[T]` is well above it.

    The exact form has no comparably short derivative, because the bound tracer depends
    on Kd as well, so its sensitivity is taken as a central difference across the tracer
    interval instead.
    """
    if not isinstance(tracer_kd, Interval):
        return 0.0
    point = _tracer_kd_point(tracer_kd)

    if receptor_conc is None:
        ratio = tracer_conc / point
        return (ratio / (1.0 + ratio)) * (tracer_kd.half_width / point)

    lower, upper = _finite(tracer_kd.lower), _finite(tracer_kd.upper)
    if lower is None or upper is None or lower <= 0:
        return 0.0
    middle = _converter(tracer_conc, point, receptor_conc)(measured_point)
    if middle <= 0:
        return 0.0
    at_low = _converter(tracer_conc, lower, receptor_conc)(measured_point)
    at_high = _converter(tracer_conc, upper, receptor_conc)(measured_point)
    return abs(at_high - at_low) / (2.0 * middle)


@overload
def ki_from_ic50(
    ic50: FitResult,
    tracer_conc: float,
    tracer_kd: float | Interval,
    receptor_conc: float | None = None,
) -> Interval: ...


@overload
def ki_from_ic50(
    ic50: Interval,
    tracer_conc: float,
    tracer_kd: float | Interval,
    receptor_conc: float | None = None,
) -> Interval: ...


@overload
def ki_from_ic50(
    ic50: float,
    tracer_conc: float,
    tracer_kd: Interval,
    receptor_conc: float | None = None,
) -> Interval: ...


@overload
def ki_from_ic50(
    ic50: float,
    tracer_conc: float,
    tracer_kd: float,
    receptor_conc: float | None = None,
) -> float: ...


def ki_from_ic50(
    ic50: float | Interval | FitResult,
    tracer_conc: float,
    tracer_kd: float | Interval,
    receptor_conc: float | None = None,
) -> float | Interval:
    """Correct a displacement IC50 into the inhibition constant Ki (Cheng-Prusoff).

    A measured IC50 is not a property of the competitor alone. Raising the
    concentration of the labelled ligand that has to be displaced raises the IC50 along
    with it, so two laboratories running the same competitor at different tracer
    concentrations get different numbers. Dividing by `1 + [tracer] / Kd_tracer`
    removes that dependence, which is what makes Ki comparable between assays.

    Pass `receptor_conc` whenever it is known. With it the exact correction is used
    instead of the standard Cheng-Prusoff form, which assumes the receptor is dilute
    enough that it neither depletes the tracer nor takes up an appreciable share of the
    competitor. Where that does not hold the standard form is biased, and the bias does
    not disappear for a weak competitor: it settles at a systematic offset set by how
    much tracer the receptor sequesters. A `FitResult` carries the value given to `fit`,
    so passing the result is enough.

    Pass the whole `FitResult` rather than one interval out of it: only the result
    carries the slope alongside the IC50 and the receptor concentration.

    Give `tracer_kd` as an `Interval` when its uncertainty is known, which it usually is
    to no better than 20%. Its contribution reaches Ki damped by `r / (1 + r)` for
    `r = [T] / Kd`, so at `r = 2.5` a 20% uncertainty there outweighs a well-measured
    IC50 several times over; treating it as exact reports an interval narrower than the
    experiment supports. `tracer_conc` and `receptor_conc` are taken as exact.

    Args:
        ic50: The fitted IC50, as a `FitResult`, as its `Interval`, or as a plain value.
            A `FitResult` is read through the model's `location` role, so it works for
            whichever model was fitted.
        tracer_conc: Total concentration of the labelled ligand, or of the substrate in
            an enzyme assay. Must be the concentration the IC50 was measured at. This is
            the total added, not the free concentration, which the exact correction
            derives from it.
        tracer_kd: Dissociation constant of that labelled ligand, or Km of the
            substrate, in the same unit as `tracer_conc`. An `Interval` folds its
            uncertainty into the result.
        receptor_conc: Total concentration of the receptor, in the same unit. Selects
            the exact correction. Taken from the `FitResult` when one is given and this
            is left as None.

    Returns:
        Ki, in the same unit as `ic50`. An `Interval` whenever either input carries one,
        otherwise a plain value. A limit that was undetermined stays undetermined, and a
        limit driven to zero or below by the combined uncertainty is reported as
        undetermined rather than as a Ki of zero, which would mean infinitely tight
        binding.

    Raises:
        ValueError: If `tracer_conc` or `receptor_conc` is negative or not finite,
            `tracer_kd` is not finite and positive, `tracer_kd` is an `Interval` with an
            undetermined limit, the receptor leaves no free tracer, or the corrected Ki
            is not positive.

    Warns:
        UserWarning: If `receptor_conc` is unavailable, since the bias of the standard
            form then cannot be assessed.
        UserWarning: If the fitted slope's interval excludes 1, which puts the result
            outside what either form describes. The value is still returned, because the
            modified forms that cover a slope away from 1 do not agree with one another
            and picking one is the caller's decision, not this library's.

    Note:
        Both forms assume the competitor and the tracer exclude each other from a single
        site. That fails for allosteric competition, and no fit can detect the
        difference: the curve looks the same. The same expression covers competitive
        enzyme inhibition, with `[S]` and `Km` in place of the tracer; a non-competitive
        or uncompetitive mechanism needs a different correction and this one does not
        apply.

        Combining the two uncertainties is first order and assumes they are independent,
        which they are when the tracer constant comes from a separate experiment.
    """
    if not math.isfinite(tracer_conc) or tracer_conc < 0:
        raise ValueError(f"tracer_conc must be finite and non-negative, got {tracer_conc!r}")

    if isinstance(ic50, FitResult):
        _check_slope(ic50)
        if receptor_conc is None:
            receptor_conc = ic50.receptor_conc
        measured: float | Interval = ic50.intervals[ic50.model.location]
    else:
        measured = ic50

    if receptor_conc is not None and (not math.isfinite(receptor_conc) or receptor_conc < 0):
        raise ValueError(f"receptor_conc must be finite and non-negative, got {receptor_conc!r}")

    kd_point = _tracer_kd_point(tracer_kd)
    if receptor_conc is None:
        _warn_approximate(tracer_conc, kd_point)

    convert = _converter(tracer_conc, kd_point, receptor_conc)
    point, down, up = _map_limits(measured, convert)
    tracer_rel = _tracer_sensitivity(
        measured.point if isinstance(measured, Interval) else measured,
        tracer_conc,
        tracer_kd,
        receptor_conc,
    )

    if not isinstance(measured, Interval) and tracer_rel == 0.0:
        return point

    method: Method = measured.method if isinstance(measured, Interval) else "asymptotic"
    return _combine(point, down, up, tracer_rel, method)


def _combine(
    point: float,
    down: float | None,
    up: float | None,
    tracer_rel: float,
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
        widened = math.hypot(down, tracer_rel)
        lower = point * (1.0 - widened) if widened < 1.0 else None
    if up is not None:
        upper = point * (1.0 + math.hypot(up, tracer_rel))
    # `method` describes where the IC50 interval came from; the tracer term is folded into it.
    return Interval(point=point, lower=lower, upper=upper, method=method)


def _warn_approximate(tracer_conc: float, tracer_kd: float) -> None:
    """Say that the Cheng-Prusoff bias could not be assessed.

    The bias comes from the receptor sequestering tracer, so bounding it needs the total
    receptor concentration. Without that value there is nothing to check against, and
    the size of the error cannot be inferred from the tracer terms alone.
    """
    warnings.warn(
        "receptor_conc was not supplied, so the standard Cheng-Prusoff form was used and "
        "its bias could not be assessed. That form assumes the receptor is dilute against "
        "the tracer and its constant; where it is not, the error does not vanish for a weak "
        "competitor but settles at a systematic offset. Pass receptor_conc= to use the exact "
        "correction instead.",
        UserWarning,
        stacklevel=3,
    )


def _check_slope(result: FitResult) -> None:
    """Warn when the fitted slope leaves the standard Cheng-Prusoff form behind."""
    exponent = result.model.exponent
    if exponent is None or exponent not in result.intervals:
        return
    slope = result.intervals[exponent]
    # A zero-width interval carries no spread to test against, so whether it happens to bracket 1 comes
    # down to rounding in the last place. Warning off that would fire on data the model fits exactly.
    if not slope.bounded or slope.zero_width or slope.contains(1.0):
        return
    depletion = (
        " The receptor concentration was not supplied, so depletion cannot be ruled out as the cause."
        if result.receptor_conc is None
        else ""
    )
    warnings.warn(
        f"{result.model.label(exponent)} = {slope.format()} has a confidence interval that excludes 1. "
        "The correction assumes a single site, mutually exclusive competition and a slope of 1, so it "
        "does not hold here. Ligand depletion steepens a displacement curve and is the most common "
        "cause: check the receptor concentration against the tracer constant before adopting a "
        "modified form that raises the terms to powers." + depletion,
        UserWarning,
        stacklevel=3,
    )
