"""Confidence intervals and uncertainty-aware formatting.

Three methods are available, and they answer different questions.

`asymptotic`
    Reads the curvature of the sum-of-squares surface off the covariance matrix at
    the optimum. Cheap, but it assumes a symmetric interval and a surface that
    closes; when a parameter is not identifiable the standard error diverges.

`profile`
    Pins the parameter at a series of values, refits everything else at each of them,
    and looks for where the residual sum of squares rises significantly above its
    minimum (an F-test). This measures the surface instead of approximating it, so
    the interval may be asymmetric and one side may be reported as undetermined,
    which is how a one-sided limit such as "Kd > 1.6" arises.

`bootstrap`
    Resamples the data and refits, giving the distribution of the estimate without
    assuming a shape for it. This matches how errors are usually reported in this
    field, from repeated experiments. With replicate measurements the resampling is
    over replicates; without them it falls back to resampling residuals.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Method = Literal["asymptotic", "profile", "bootstrap"]


def _finite(value: float | None) -> float | None:
    """The value itself when it is a usable limit, otherwise None."""
    if value is None or not math.isfinite(value):
        return None
    return value


# Fewest successful bootstrap samples needed to form a percentile interval. The 2.5 / 97.5 percentiles are
# estimated from the tails, so with fewer samples than this the ends of the interval rest on a few points.
MIN_BOOTSTRAP_SAMPLES = 100

# How many multiples of the measured range to search over. A half-saturation constant beyond 1000 times
# the highest concentration cannot be distinguished from an infinite one by any amount of computation.
SEARCH_SPAN = 1e3


@dataclass(frozen=True)
class Interval:
    """A confidence interval that may be asymmetric or one-sided.

    Attributes:
        point: The point estimate.
        lower: Lower limit, or None when the parameter is unbounded below.
        upper: Upper limit, or None when the parameter is unbounded above.
        method: Which method produced the interval.
    """

    point: float
    lower: float | None
    upper: float | None
    method: Method = "asymptotic"

    @property
    def bounded(self) -> bool:
        """True when both limits exist and are finite.

        An infinite limit carries no more information than a missing one, so both are
        undetermined. Treating infinity as a limit would let it slip past the checks
        that exist for None.
        """
        return _finite(self.lower) is not None and _finite(self.upper) is not None

    @property
    def half_width(self) -> float:
        """Half of the interval width, or infinity when a limit is missing.

        Provided for the common case of a roughly symmetric interval. Prefer
        `lower` and `upper` when the interval may be skewed.
        """
        lower, upper = _finite(self.lower), _finite(self.upper)
        if lower is None or upper is None:
            return math.inf
        return (upper - lower) / 2.0

    @property
    def zero_width(self) -> bool:
        """True when both limits exist, are finite, and coincide.

        With residuals of exactly zero there is no scatter to estimate a spread from,
        so the interval collapses. Reported as such rather than as a bare number,
        which would read as an uncertainty that was never assessed.
        """
        lower, upper = _finite(self.lower), _finite(self.upper)
        return lower is not None and lower == upper

    @property
    def symmetric(self) -> bool:
        """True when the two sides differ by less than a tenth of the width."""
        lower, upper = _finite(self.lower), _finite(self.upper)
        if lower is None or upper is None:
            return False
        up, down = upper - self.point, self.point - lower
        span = max(up, down)
        return span <= 0 or abs(up - down) / span < 0.1

    def contains(self, value: float) -> bool:
        """Whether a value lies inside the interval, treating None as unbounded."""
        lower, upper = _finite(self.lower), _finite(self.upper)
        if lower is not None and value < lower:
            return False
        return not (upper is not None and value > upper)

    def format(self, unit: str = "") -> str:
        """Render the estimate with a precision justified by its uncertainty."""
        suffix = f" {unit}" if unit else ""
        lower, upper = _finite(self.lower), _finite(self.upper)
        if upper is None:
            if lower is None:
                # The point estimate only stopped somewhere along a flat valley, so it is shown to few digits.
                return f"{self.point:.2g}{suffix} (both limits undetermined)"
            return f"> {_sig(lower, 2)}{suffix} (95%, upper limit undetermined)"
        if lower is None:
            return f"< {_sig(upper, 2)}{suffix} (95%, lower limit undetermined)"
        if self.zero_width:
            # A fit whose residuals are exactly 0. A bare number would read as "uncertainty unknown", so the
            # reason the width is 0 is spelled out too.
            return f"{_significant(self.point, 4)}{suffix} +/- 0 (no residual scatter)"
        if self.symmetric:
            return format_with_uncertainty(self.point, self.half_width, unit)
        return self._format_asymmetric(lower, upper, suffix)

    def _format_asymmetric(self, lower: float, upper: float, suffix: str) -> str:
        """Render a skewed interval as `point [lower, upper]`.

        Two regimes are needed. For a moderately skewed interval the three numbers
        share a decimal place, since rounding each separately can collapse the
        interval into something like [1.1, 1.1]. A shared decimal place cannot
        represent endpoints that differ by orders of magnitude, though, which is the
        normal shape for a concentration constant: rounding to the narrow side would
        print a positive lower limit as 0, and a Kd of zero means infinitely tight
        binding.
        """
        if _spans_orders_of_magnitude(lower, upper):
            return self._format_by_significant_digits(lower, upper, suffix)

        reference = min(upper - self.point, self.point - lower)
        if reference <= 0 or not np.isfinite(reference):
            reference = (upper - lower) / 2.0
        decimals = _decimals_for(reference)
        if any(value != 0 and round(value, decimals) == 0 for value in (self.point, lower, upper)):
            # Catches the case where the decimal place is too coarse and a non-zero value would print as 0.
            return self._format_by_significant_digits(lower, upper, suffix)
        return f"{_fixed(self.point, decimals)}{suffix} [{_fixed(lower, decimals)}, {_fixed(upper, decimals)}] (95% CI)"

    def _format_by_significant_digits(self, lower: float, upper: float, suffix: str) -> str:
        # Over an interval spanning several orders of magnitude the point estimate is meaningful to 2 digits only.
        return f"{_significant(self.point, 2)}{suffix} [{_significant(lower, 2)}, {_significant(upper, 2)}] (95% CI)"


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


def profile_bounds(
    ssr_at: Callable[[float], float | None],
    best: float,
    ssr_min: float,
    threshold: float,
    search_lower: float,
    search_upper: float,
    log_scale: bool,
    step: float = 1.0,
    max_steps: int = 200,
    bisect_steps: int = 40,
    on_direction_start: Callable[[], None] | None = None,
) -> tuple[float | None, float | None]:
    """Find where the residual sum of squares crosses `threshold` on each side.

    The search is confined to `[search_lower, search_upper]`, which must be chosen
    from the scale of the data rather than from the point estimate. When a parameter
    is not identifiable the estimate itself wanders along a plateau and can stop at
    an arbitrary value, so a range defined relative to it would be meaningless. A
    Kd a thousand times beyond the highest measured concentration, for instance,
    cannot be distinguished from an infinite one by any amount of computation.

    Args:
        ssr_at: Refits everything else with the parameter pinned at the given value
            and returns the residual sum of squares, or None when that refit could
            not be carried out. Returning None rather than infinity matters: an
            infinite value is indistinguishable from "above the threshold" and would
            be read as a crossing, inventing a limit that was never established.
        best: The unconstrained estimate.
        ssr_min: Residual sum of squares at the optimum.
        threshold: The value of the sum of squares that marks the interval edge.
        search_lower: Lowest value to examine.
        search_upper: Highest value to examine.
        log_scale: Step multiplicatively rather than additively. Use for parameters
            that must stay positive and can span orders of magnitude.
        step: Initial additive step, used when `log_scale` is False.
        max_steps: Give up expanding after this many steps.
        bisect_steps: Iterations used to refine the crossing.
        on_direction_start: Called before each direction is walked. A caller that
            warm-starts `ssr_at` from the previous solution uses this to put that
            state back to the unconstrained optimum, so that the upper limit does not
            depend on where the downward walk finished.

    Returns:
        `(lower, upper)`, with None on a side where the surface never rises enough
        within the search range, or where the search could not be started; either way
        that limit is undetermined. When an evaluation fails after the crossing has
        been bracketed, the outer end of the bracket is returned, erring towards a
        wider interval.
    """
    if not np.isfinite(threshold) or threshold <= ssr_min:
        return best, best

    def walk(direction: int) -> float | None:
        if on_direction_start is not None:
            on_direction_start()
        limit = search_upper if direction > 0 else search_lower
        if (direction > 0 and best >= limit) or (direction < 0 and best <= limit):
            return None  # The point estimate already lies outside the search range; that side is undetermined.

        inside = best
        factor = 1.6
        current_step = step
        for _ in range(max_steps):
            if log_scale and inside > 0:
                candidate = inside * (factor if direction > 0 else 1.0 / factor)
            else:
                candidate = inside + direction * current_step
                current_step *= factor
            reached_limit = (direction > 0 and candidate >= limit) or (direction < 0 and candidate <= limit)
            if reached_limit:
                candidate = limit

            value = ssr_at(candidate)
            if value is None:
                return None  # The evaluation failed. There is no basis for asserting a limit.
            if value > threshold:
                # inside is <= threshold, outside is > threshold. Bracket the two and converge on the crossing.
                inside_v, outside_v = inside, candidate
                for _ in range(bisect_steps):
                    if log_scale and inside_v > 0 and outside_v > 0:
                        mid = math.sqrt(inside_v * outside_v)
                    else:
                        mid = (inside_v + outside_v) / 2.0
                    mid_value = ssr_at(mid)
                    if mid_value is None:
                        # Refinement stalled. Returning the inner end would narrow the interval, so return the outer.
                        return outside_v
                    if mid_value > threshold:
                        outside_v = mid
                    else:
                        inside_v = mid
                    if log_scale and inside_v > 0 and outside_v > 0:
                        converged = abs(math.log(outside_v) - math.log(inside_v)) <= 1e-10
                    else:
                        converged = abs(outside_v - inside_v) <= 1e-10 * max(abs(inside_v), 1.0)
                    if converged:
                        break
                return inside_v

            if reached_limit:
                return None  # The threshold is not exceeded out to the edge of the search range.
            inside = candidate
        return None

    return walk(-1), walk(+1)


def percentile_interval(samples: NDArray[np.float64], point: float) -> Interval:
    """Build a percentile interval from bootstrap samples."""
    finite = samples[np.isfinite(samples)]
    if finite.size < MIN_BOOTSTRAP_SAMPLES:
        return Interval(point=point, lower=None, upper=None, method="bootstrap")
    lo, hi = (float(v) for v in np.percentile(finite, [2.5, 97.5]))
    return Interval(point=point, lower=lo, upper=hi, method="bootstrap")
