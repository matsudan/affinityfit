"""Number formatting that reports a precision justified by the uncertainty.

Reporting `Kd = 4.70e-8` from data that only constrain it to 17% overstates the
result. The helpers here decide how many digits a point estimate or an interval
endpoint gets to keep, from the size of the uncertainty rather than from a fixed
number of decimal places.
"""

from __future__ import annotations

import math

import numpy as np


def _spans_orders_of_magnitude(lower: float, upper: float, factor: float = 100.0) -> bool:
    """Whether the limits are far enough apart that a shared decimal place fails.

    Only meaningful when both limits sit on the same side of zero; an interval that
    straddles zero has no ratio to speak of.
    """
    if lower == 0.0 or upper == 0.0 or (lower < 0) != (upper < 0):
        return False
    return abs(upper / lower) > factor or abs(lower / upper) > factor


def _significant(value: float, digits: int) -> str:
    """Round to a number of significant digits, avoiding exponent notation when readable.

    `f"{400:.2g}"` gives `4e+02`, which is hard to read next to other numbers in the
    same interval. Plain decimal notation is used while the exponent stays within a
    comfortable range, and the value is rounded first so that the digits shown are
    the ones that are meant.
    """
    if value == 0.0 or not np.isfinite(value):
        return f"{value:.{digits}g}"
    exponent = math.floor(math.log10(abs(value)))
    rounded = round(value, -(exponent - (digits - 1)))
    if -4 <= exponent < 6:
        return f"{rounded:.{max(0, digits - 1 - exponent)}f}"
    return f"{rounded:.{digits - 1}e}"


def _decimals_for(unc: float) -> int:
    """Decimal places justified by an uncertainty of this size."""
    if not np.isfinite(unc) or unc <= 0:
        return 3
    digits = _uncertainty_digits(unc)
    exponent = math.floor(math.log10(abs(unc)))
    return -(exponent - (digits - 1))


def _fixed(value: float, decimals: int) -> str:
    """Format at a given decimal place, switching to scientific notation when extreme."""
    if value != 0 and not (1e-4 <= abs(value) < 1e6):
        return f"{value:.2e}"
    return f"{value:.{max(0, decimals)}f}"


def _sig(value: float, digits: int) -> str:
    return f"{value:.{digits}g}"


def _uncertainty_digits(unc: float) -> int:
    """Significant digits to quote for an uncertainty: 2 if it starts with 1 or 2.

    This is the usual convention: quoting more digits of an uncertainty than it
    carries is meaningless, but rounding 1.4 down to 1 loses a third of it.
    """
    exponent = math.floor(math.log10(abs(unc)))
    mantissa = abs(unc) / 10.0**exponent
    return 2 if mantissa < 3.0 else 1


def format_with_uncertainty(point: float, unc: float, unit: str = "") -> str:
    """Format `point +/- unc`, rounding both to a precision the uncertainty supports.

    Reporting `Kd = 4.70e-8` from data that only constrain it to 17% overstates the
    result; this returns `(4.7 +/- 0.8)e-08` instead.

    Args:
        point: The point estimate.
        unc: Half-width of the interval. Must be positive to trigger rounding.
        unit: Appended to the result when given.

    Returns:
        A formatted string.
    """
    suffix = f" {unit}" if unit else ""
    if not np.isfinite(unc) or unc <= 0 or not np.isfinite(point):
        return f"{point:.4g}{suffix}"

    digits = _uncertainty_digits(unc)
    exponent = math.floor(math.log10(abs(unc)))
    decimals = -(exponent - (digits - 1))
    unc_r = round(unc, decimals)
    if unc_r > 0 and _uncertainty_digits(unc_r) != digits:
        # When rounding carries into the next digit, as 0.999 -> 1.0 does, decide the digit count again.
        digits = _uncertainty_digits(unc_r)
        exponent = math.floor(math.log10(abs(unc_r)))
        decimals = -(exponent - (digits - 1))
        unc_r = round(unc, decimals)
    point_r = round(point, decimals)

    # With a large exponent, use a notation that shares the exponent: (4.7 +/- 0.8)e-08
    if point_r != 0 and not (1e-4 <= abs(point_r) < 1e5):
        shift = math.floor(math.log10(abs(point_r)))
        scale = 10.0**shift
        places = max(0, decimals + shift)
        return f"({point_r / scale:.{places}f} +/- {unc_r / scale:.{places}f})e{shift:+03d}{suffix}"

    places = max(0, decimals)
    return f"{point_r:.{places}f} +/- {unc_r:.{places}f}{suffix}"
