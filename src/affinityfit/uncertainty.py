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
from dataclasses import dataclass
from typing import Literal

import numpy as np

from affinityfit.formatting import (
    _decimals_for,
    _fixed,
    _sig,
    _significant,
    _spans_orders_of_magnitude,
    format_with_uncertainty,
)

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
