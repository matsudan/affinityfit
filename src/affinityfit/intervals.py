"""Confidence-interval strategies: asymptotic, profile, and bootstrap.

Each strategy answers a different question about the same fitted problem.

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

Every strategy operates on a `FittedProblem`, the structural interface a fitted
optimisation problem must offer; it names what these strategies read from that
problem without depending on the concrete class that builds it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from affinityfit.models import Model
from affinityfit.uncertainty import MIN_BOOTSTRAP_SAMPLES, SEARCH_SPAN, Interval, Method

# Cap on the exponent when a log-scale parameter is turned back into a plain value (10**309 is not representable).
_LOG_LIMIT = 300.0


class _FittedDataset(Protocol):
    """The parts of a dataset these strategies read."""

    @property
    def conc(self) -> NDArray[np.float64]: ...

    @property
    def replicates(self) -> NDArray[np.float64] | None: ...

    @property
    def observed(self) -> NDArray[np.float64]: ...


class FittedProblem(Protocol):
    """Structural interface a fitted optimisation problem must offer.

    Naming this separately from the class that implements it is what lets the
    interval strategies below depend on the shape of a solved problem rather than
    on the layout and solving mechanics that produce one.
    """

    @property
    def model(self) -> Model: ...

    @property
    def slot_param(self) -> list[str]: ...

    @property
    def n_slots(self) -> int: ...

    @property
    def n_points(self) -> int: ...

    @property
    def lower(self) -> NDArray[np.float64]: ...

    @property
    def upper(self) -> NDArray[np.float64]: ...

    @property
    def log_slots(self) -> list[bool]: ...

    @property
    def point_weights(self) -> list[NDArray[np.float64]]: ...

    @property
    def datasets(self) -> Sequence[_FittedDataset]: ...

    def unpack(self, x: NDArray[np.float64], index: int) -> tuple[float, ...]: ...

    def solve(
        self,
        x_start: NDArray[np.float64] | None = None,
        pinned: Mapping[int, float] | None = None,
        signals: Sequence[NDArray[np.float64]] | None = None,
    ) -> tuple[NDArray[np.float64], float, NDArray[np.float64] | None]: ...


def _jacobian_rank(jac: NDArray[np.float64]) -> int:
    """Numerical rank of the Jacobian, computed after normalising the columns.

    Without normalisation the rank test is dominated by whichever parameter happens
    to have the largest derivative, which for a logarithmically fitted concentration
    constant can be many orders of magnitude away from the others.
    """
    norms = np.linalg.norm(jac, axis=0)
    scaled = jac / np.where(norms > 0.0, norms, 1.0)
    return int(np.linalg.matrix_rank(scaled, tol=1e-8))


def _undetermined(x: NDArray[np.float64], method: Method) -> list[Interval]:
    return [Interval(point=float(v), lower=None, upper=None, method=method) for v in x]


def asymptotic_intervals(
    problem: FittedProblem,
    x: NDArray[np.float64],
    ssr: float,
    jac: NDArray[np.float64],
) -> list[Interval]:
    """Intervals from the curvature of the sum-of-squares surface at the optimum.

    The Jacobian is with respect to the optimiser's variables, so for a parameter
    fitted as a logarithm the interval is formed in the logarithm and then mapped
    back by exponentiation. That keeps the result multiplicative, which both matches
    how affinities are reported and guarantees a positive lower limit for a
    concentration constant.

    Two situations make the covariance meaningless, and both yield undetermined
    intervals rather than a number.

    - No degrees of freedom. With as many parameters as points the curve passes
      through every point by construction, leaving no residual variance to estimate a
      spread from.
    - A rank-deficient Jacobian. `pinv` discards the singular directions without
      complaint, so a parameter the data cannot pin down would come back with a tiny
      interval.
    """
    dof = problem.n_points - problem.n_slots
    if dof < 1 or _jacobian_rank(jac) < problem.n_slots:
        return _undetermined(x, "asymptotic")

    try:
        jtj_inv = np.linalg.pinv(jac.T @ jac)
        stderr = np.sqrt(np.clip(np.diag(jtj_inv) * ssr / dof, 0.0, np.inf))
    except np.linalg.LinAlgError:  # pragma: no cover - numerically rare
        stderr = np.full(problem.n_slots, np.nan)
    half = float(stats.t.ppf(0.975, dof)) * stderr

    intervals: list[Interval] = []
    for j in range(problem.n_slots):
        value = float(x[j])
        # A half-width too large to represent as an exponent means the curvature is near 0, a sign that the parameter
        # is unidentifiable, and exponentiating it gives inf. An infinite limit is no limit, so return undetermined.
        if not np.isfinite(half[j]) or (problem.log_slots[j] and half[j] >= _LOG_LIMIT):
            intervals.append(Interval(point=value, lower=None, upper=None, method="asymptotic"))
            continue
        if problem.log_slots[j]:
            lower, upper = value * 10.0 ** (-half[j]), value * 10.0 ** (+half[j])
        else:
            lower, upper = value - half[j], value + half[j]
        if not (np.isfinite(lower) and np.isfinite(upper)):
            intervals.append(Interval(point=value, lower=None, upper=None, method="asymptotic"))
            continue
        intervals.append(Interval(point=value, lower=float(lower), upper=float(upper), method="asymptotic"))
    return intervals


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
                        mid = float(np.sqrt(inside_v * outside_v))
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
                        converged = abs(np.log(outside_v) - np.log(inside_v)) <= 1e-10
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


def profile_intervals(problem: FittedProblem, x: NDArray[np.float64], ssr: float) -> list[Interval]:
    dof = problem.n_points - problem.n_slots
    if dof < 1:
        return [Interval(point=float(v), lower=None, upper=None, method="profile") for v in x]
    threshold = ssr * (1.0 + float(stats.f.ppf(0.95, 1, dof)) / dof)

    all_conc = np.concatenate([d.conc for d in problem.datasets])
    positive_conc = all_conc[all_conc > 0]
    conc_max = float(positive_conc.max()) if positive_conc.size else 1.0
    conc_min = float(positive_conc.min()) if positive_conc.size else 1e-6
    signal_scale = float(max(np.abs(np.concatenate([d.observed for d in problem.datasets])).max(), 1e-12))

    intervals: list[Interval] = []
    for j in range(problem.n_slots):
        param = problem.slot_param[j]
        # The search range comes from the scale of the measured data, so a diverged point estimate does not affect it.
        if param == problem.model.location:
            search_lo, search_hi = conc_min / SEARCH_SPAN, conc_max * SEARCH_SPAN
        elif param in (problem.model.amplitude, problem.model.baseline):
            # Amplitude and baseline carry a sign, so the negative side is searched too.
            search_lo, search_hi = -signal_scale * SEARCH_SPAN, signal_scale * SEARCH_SPAN
        else:
            search_lo, search_hi = problem.lower[j], problem.upper[j]
        # Clip at the model's own bounds, such as Vmax >= 0 for michaelis.
        search_lo = max(search_lo, float(problem.lower[j]))
        search_hi = min(search_hi, float(problem.upper[j]))

        # Each step outward restarts from the previous solution, because re-solving from the optimum every time makes
        # the inner optimisation more likely to fail. When the direction changes, reset to the optimum so that the
        # upper limit does not depend on the search that produced the lower one.
        state = {"x": x}

        def ssr_at(value: float, slot: int = j, state: dict = state) -> float | None:
            # When the inner refit does not converge, mark that direction as undetermined rather than failing the
            # whole fit. Substituting inf would be indistinguishable from exceeding the threshold.
            try:
                solved, ssr_value, _ = problem.solve(x_start=state["x"], pinned={slot: value})
            except (RuntimeError, ValueError):
                return None
            if not np.isfinite(ssr_value):
                return None
            state["x"] = solved
            return ssr_value

        def reset_warm_start(state: dict = state, start: NDArray[np.float64] = x) -> None:
            state["x"] = start

        lo, hi = profile_bounds(
            ssr_at,
            best=float(x[j]),
            ssr_min=ssr,
            threshold=threshold,
            search_lower=search_lo,
            search_upper=search_hi,
            # Use the value the problem holds rather than rebuilding the test from the bounds (defined in one place).
            log_scale=problem.log_slots[j],
            step=max(abs(float(x[j])) * 0.1, signal_scale * 1e-3),
            on_direction_start=reset_warm_start,
        )
        intervals.append(Interval(point=float(x[j]), lower=lo, upper=hi, method="profile"))
    return intervals


def percentile_interval(samples: NDArray[np.float64], point: float) -> Interval:
    """Build a percentile interval from bootstrap samples."""
    finite = samples[np.isfinite(samples)]
    if finite.size < MIN_BOOTSTRAP_SAMPLES:
        return Interval(point=point, lower=None, upper=None, method="bootstrap")
    lo, hi = (float(v) for v in np.percentile(finite, [2.5, 97.5]))
    return Interval(point=point, lower=lo, upper=hi, method="bootstrap")


def bootstrap_intervals(
    problem: FittedProblem,
    x: NDArray[np.float64],
    jac: NDArray[np.float64],
    n_boot: int,
    seed: int,
) -> tuple[list[Interval], int]:
    """Percentile intervals from resampled data, plus the number of failed resamples.

    Resamples on which the fit does not converge are counted, not merely dropped.
    Convergence is not independent of the data: the resamples that succeed are the
    ones that were easy to fit, so discarding the rest biases the interval narrow.

    Each resample is refit from the point estimate `x`. Along a rank-deficient
    direction that point sits on a ridge of equally good fits, so every resample
    would converge back to nearly the same point regardless of which data were
    drawn, understating the true uncertainty. This case is therefore diagnosed as
    undetermined up front, the same as in `asymptotic_intervals`.
    """
    if problem.n_points - problem.n_slots < 1:
        # With no degrees of freedom the residuals are identically 0, so every resample returns the same answer.
        return _undetermined(x, "bootstrap"), 0
    if _jacobian_rank(jac) < problem.n_slots:
        return _undetermined(x, "bootstrap"), 0

    rng = np.random.default_rng(seed)
    fitted = [problem.model(d.conc, *problem.unpack(x, i)) for i, d in enumerate(problem.datasets)]
    # Residuals are standardised by sigma before resampling and then multiplied by the per-point sigma again. Mixing
    # the raw residuals would lose the structure of the error varying in size from point to point.
    standardized = [(d.observed - fitted[i]) * problem.point_weights[i] for i, d in enumerate(problem.datasets)]

    samples = np.full((n_boot, problem.n_slots), np.nan)
    failures = 0
    for b in range(n_boot):
        signals: list[NDArray[np.float64]] = []
        for i, d in enumerate(problem.datasets):
            if d.replicates is not None:
                pick = rng.integers(0, d.replicates.shape[0], size=d.conc.size)
                signals.append(d.replicates[pick, np.arange(d.conc.size)])
            else:
                drawn = rng.choice(standardized[i], size=d.conc.size, replace=True)
                signals.append(fitted[i] + drawn / problem.point_weights[i])
        try:
            samples[b] = problem.solve(x_start=x, signals=signals)[0]
        except (RuntimeError, ValueError):
            failures += 1

    successes = n_boot - failures
    if successes < MIN_BOOTSTRAP_SAMPLES:
        return _undetermined(x, "bootstrap"), failures
    return [percentile_interval(samples[:, j], float(x[j])) for j in range(problem.n_slots)], failures
