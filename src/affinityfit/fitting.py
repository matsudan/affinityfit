"""Fitting: one dataset, or several at once with parameters shared or held constant.

Each parameter can be treated in one of three ways.

- **free** (default): estimated separately for every dataset
- **shared**: a single value estimated across all datasets
- **fixed**: held at a constant and not estimated

Sharing is decisive when the measured range of a dataset does not bracket its
half-saturation constant. Such a curve only determines the ratio of amplitude to
half-saturation constant, so neither value is identifiable on its own; sharing the
amplitude with a well-sampled curve breaks the degeneracy.

The same operation appears in the literature as global fitting, global analysis,
simultaneous analysis, or shared-parameter fitting. In Biacore terminology a shared
parameter is called "global" and a per-dataset one "local". This is unrelated to
global optimisation in the numerical sense of locating a global minimum.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import least_squares

from affinityfit.core import Diagnostic, FitResult, Statistic, _diagnose_coded, _diagnostic, _reject_non_finite
from affinityfit.models import Model, langmuir
from affinityfit.uncertainty import (
    MIN_BOOTSTRAP_SAMPLES,
    SEARCH_SPAN,
    Interval,
    Method,
    percentile_interval,
    profile_bounds,
)

# Cap on the exponent when a log-scale parameter is turned back into a plain value (10**309 is not representable).
_LOG_LIMIT = 300.0


@dataclass(frozen=True, eq=False)
class Dataset:
    """One titration dataset passed to a fit.

    Comparison is by identity. Value equality is not offered because it has no
    unambiguous meaning for float arrays, and the generated `__eq__` and `__hash__`
    would raise on ndarray fields.

    Args:
        name: Label used for display, for example "oxidized".
        conc: Ligand concentration.
        signal: Observed signal. May be omitted when `replicates` is given, in which
            case the mean over the replicates is used.
        receptor_conc: Concentration of the fixed partner, used for the
            ligand-depletion diagnostic.
        replicates: Optional array of shape `(n_replicates, n_points)` holding the
            individual repeat measurements. Enables bootstrap resampling over
            replicates rather than over residuals.
        sigma: Optional per-point standard deviations. Without them every point
            counts equally, which assumes the measurement error is the same size
            everywhere. That holds for a response in resonance units but not for
            fluorescence, luminescence or absorbance, where the error grows with the
            signal; on such data an unweighted fit is both less precise and prone to
            intervals that are too narrow. Only the relative sizes matter, since the
            overall scale is still estimated from the residuals. In a fit over several
            datasets, sigma must be given for all of them or for none.

    Raises:
        ValueError: If neither signal nor replicates is given, the arrays are
            inconsistent, fewer than 2 points are given, a concentration is negative,
            any value is NaN or infinite, or a sigma is not strictly positive.
    """

    name: str
    conc: NDArray[np.float64]
    signal: NDArray[np.float64] | None = None
    receptor_conc: float | None = None
    replicates: NDArray[np.float64] | None = None
    sigma: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        conc = np.asarray(self.conc, dtype=float)
        _reject_non_finite(conc, "concentration", self.name)
        if conc.size < 2:
            raise ValueError(f"{self.name}: only {conc.size} data point(s) found.")
        if np.any(conc < 0):
            raise ValueError(f"{self.name}: concentration contains negative values.")
        object.__setattr__(self, "conc", conc)

        # Validate replicates before signal, because signal is filled from their mean when it was omitted.
        if self.replicates is not None:
            reps = np.asarray(self.replicates, dtype=float)
            if reps.ndim != 2 or reps.shape[1] != conc.size:
                raise ValueError(
                    f"{self.name}: replicates must have shape (n_replicates, {conc.size}), got {reps.shape}"
                )
            if reps.shape[0] < 2:
                raise ValueError(f"{self.name}: replicates need at least 2 rows, got {reps.shape[0]}")
            _reject_non_finite(reps.ravel(), "replicates", self.name)
            object.__setattr__(self, "replicates", reps)

        if self.signal is None:
            if self.replicates is None:
                raise ValueError(f"{self.name}: give either signal or replicates.")
            object.__setattr__(self, "signal", np.asarray(self.replicates, dtype=float).mean(axis=0))
        else:
            signal = np.asarray(self.signal, dtype=float)
            if signal.shape != conc.shape:
                raise ValueError(f"{self.name}: conc and signal have different lengths ({conc.size} vs {signal.size})")
            # Catch non-finite values here, before they reach scipy's internals and raise an error that hides the cause.
            _reject_non_finite(signal, "signal", self.name)
            object.__setattr__(self, "signal", signal)

        if self.receptor_conc is not None and not math.isfinite(self.receptor_conc):
            raise ValueError(f"{self.name}: receptor_conc is not finite.")

        if self.sigma is not None:
            sigma = np.asarray(self.sigma, dtype=float)
            if sigma.shape != conc.shape:
                raise ValueError(f"{self.name}: sigma must have the same shape as conc, got {sigma.shape}")
            _reject_non_finite(sigma, "sigma", self.name)
            if np.any(sigma <= 0):
                raise ValueError(f"{self.name}: sigma must be strictly positive at every point.")
            object.__setattr__(self, "sigma", sigma)

    @property
    def observed(self) -> NDArray[np.float64]:
        """The signal as an array.

        `signal` is optional at construction because it may be derived from the
        replicates, but validation always leaves one in place. This property is what
        the fitting code reads, so it does not have to carry the optionality.
        """
        signal = self.signal
        if signal is None:  # pragma: no cover - __post_init__ always fills it in
            raise ValueError(f"{self.name}: signal was not set.")
        return signal

    @property
    def weights(self) -> NDArray[np.float64]:
        """Reciprocal of `sigma`, or ones when no sigma was given."""
        if self.sigma is None:
            return np.ones_like(self.conc)
        return 1.0 / self.sigma


class _Problem:
    """Internal bookkeeping for a fit: parameter layout, residuals, and re-solving.

    Holding this separately is what lets the profile-likelihood and bootstrap code
    re-solve the very same problem with one parameter pinned, or with resampled
    signals, without duplicating the layout logic.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        model: Model,
        shared: tuple[str, ...],
        fixed: Mapping[str, float],
    ) -> None:
        self.datasets = list(datasets)
        self.model = model
        self.shared = shared
        self.fixed = dict(fixed)
        n_sets = len(self.datasets)

        # A fixed parameter gets no slot, a shared one a single slot, a free one one slot per dataset.
        self.slots: dict[str, list[int]] = {}
        self.slot_param: list[str] = []
        self.slot_dataset: list[str | None] = []
        x0: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        guesses = [model.initial(d.conc, d.observed) for d in self.datasets]

        for name in model.params:
            if name in self.fixed:
                continue
            values = [g[name] for g in guesses]
            if name in shared:
                self.slots[name] = [len(x0)]
                x0.append(float(np.median(values)))
                lower.append(model.lower(name))
                upper.append(model.upper(name))
                self.slot_param.append(name)
                self.slot_dataset.append(None)
            else:
                self.slots[name] = list(range(len(x0), len(x0) + n_sets))
                x0.extend(float(v) for v in values)
                lower.extend([model.lower(name)] * n_sets)
                upper.extend([model.upper(name)] * n_sets)
                self.slot_param.extend([name] * n_sets)
                self.slot_dataset.extend(d.name for d in self.datasets)

        self.x0 = np.clip(np.asarray(x0, dtype=float), lower, upper)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.n_slots = len(x0)
        self.n_points = int(sum(d.conc.size for d in self.datasets))
        # Which slots are handled on a logarithmic scale. `Model.is_log_scale` is the single definition, and both
        # the change of variable and the way the profile likelihood steps read it from here.
        self.log_slots = [model.is_log_scale(name) for name in self.slot_param]

        # Weights are relative, so they are normalised to a geometric mean of 1 taken over all datasets together.
        # Without that, multiplying sigma by a constant changes the absolute size of the residuals, and the answer
        # moves with it through the optimiser's convergence test. The relative weighting between datasets has to be
        # preserved, so the normalisation applies one factor to the whole problem rather than one per dataset.
        raw = np.concatenate([d.weights for d in self.datasets])
        scale = float(np.exp(np.mean(np.log(raw))))
        self.point_weights = [d.weights / scale for d in self.datasets]

    def _to_internal(self, x: NDArray[np.float64], slots: Sequence[int]) -> NDArray[np.float64]:
        """Map original-space values of the given slots into optimiser space."""
        out = np.array([x[j] for j in slots], dtype=float)
        for position, j in enumerate(slots):
            if self.log_slots[j]:
                out[position] = np.log10(max(float(x[j]), float(self.lower[j])))
        return out

    def _from_internal(self, values: NDArray[np.float64], slots: Sequence[int]) -> NDArray[np.float64]:
        """Map optimiser-space values of the given slots back to original space."""
        out = np.array(values, dtype=float)
        for position, j in enumerate(slots):
            if self.log_slots[j]:
                out[position] = 10.0 ** float(np.clip(values[position], -_LOG_LIMIT, _LOG_LIMIT))
        return out

    def _internal_bounds(self, slots: Sequence[int]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = np.array([self.lower[j] for j in slots], dtype=float)
        upper = np.array([self.upper[j] for j in slots], dtype=float)
        for position, j in enumerate(slots):
            if self.log_slots[j]:
                # An infinite upper bound overflows the moment it is exponentiated, so it is held down.
                lower[position] = max(np.log10(self.lower[j]), -_LOG_LIMIT)
                bound = np.log10(self.upper[j]) if np.isfinite(self.upper[j]) else _LOG_LIMIT
                upper[position] = min(bound, _LOG_LIMIT)
        return lower, upper

    def slot_of(self, param: str, dataset_index: int) -> int:
        pos = self.slots[param]
        return pos[0] if len(pos) == 1 else pos[dataset_index]

    def unpack(self, x: NDArray[np.float64], index: int) -> tuple[float, ...]:
        out: list[float] = []
        for name in self.model.params:
            if name in self.fixed:
                out.append(float(self.fixed[name]))
            else:
                out.append(float(x[self.slot_of(name, index)]))
        return tuple(out)

    def residual(
        self,
        x: NDArray[np.float64],
        signals: Sequence[NDArray[np.float64]] | None = None,
    ) -> NDArray[np.float64]:
        obs = signals if signals is not None else [d.observed for d in self.datasets]
        # With sigma given the residuals are weighted. Without it the weights are 1 and every point counts equally,
        # which assumes the measurement error is the same size everywhere.
        return np.concatenate(
            [
                (self.model(d.conc, *self.unpack(x, i)) - obs[i]) * self.point_weights[i]
                for i, d in enumerate(self.datasets)
            ]
        )

    def solve(
        self,
        x_start: NDArray[np.float64] | None = None,
        pinned: Mapping[int, float] | None = None,
        signals: Sequence[NDArray[np.float64]] | None = None,
    ) -> tuple[NDArray[np.float64], float, NDArray[np.float64] | None]:
        """Solve, optionally with some slots pinned and/or resampled signals.

        Values passed in and returned are always in original parameter space; the
        logarithmic change of variable is confined to this method. The Jacobian,
        however, is with respect to the optimiser's variables, which
        `_asymptotic_intervals` accounts for.

        Returns:
            `(x, ssr, jac)` where `jac` is None when slots were pinned.
        """
        pinned = dict(pinned or {})
        start = self.x0 if x_start is None else np.asarray(x_start, dtype=float)
        free = [j for j in range(self.n_slots) if j not in pinned]

        if not free:
            x = np.array([pinned.get(j, start[j]) for j in range(self.n_slots)], dtype=float)
            res = self.residual(x, signals)
            return x, float(res @ res), None

        lower, upper = self._internal_bounds(free)

        def build(values: NDArray[np.float64]) -> NDArray[np.float64]:
            x = np.empty(self.n_slots, dtype=float)
            x[free] = self._from_internal(values, free)
            for j, v in pinned.items():
                x[j] = v
            return x

        sol = least_squares(
            lambda values: self.residual(build(values), signals),
            x0=np.clip(self._to_internal(start, free), lower, upper),
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=20000,
        )
        if not sol.success:
            raise RuntimeError(f"Optimization did not converge: {sol.message}")
        return build(sol.x), float(sol.fun @ sol.fun), (sol.jac if not pinned else None)


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


def _asymptotic_intervals(
    problem: _Problem,
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


def _profile_intervals(problem: _Problem, x: NDArray[np.float64], ssr: float) -> list[Interval]:
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
            # Use the value `_Problem` holds rather than rebuilding the test from the bounds (defined in one place).
            log_scale=problem.log_slots[j],
            step=max(abs(float(x[j])) * 0.1, signal_scale * 1e-3),
            on_direction_start=reset_warm_start,
        )
        intervals.append(Interval(point=float(x[j]), lower=lo, upper=hi, method="profile"))
    return intervals


def _bootstrap_intervals(
    problem: _Problem,
    x: NDArray[np.float64],
    n_boot: int,
    seed: int,
) -> tuple[list[Interval], int]:
    """Percentile intervals from resampled data, plus the number of failed resamples.

    Resamples on which the fit does not converge are counted, not merely dropped.
    Convergence is not independent of the data: the resamples that succeed are the
    ones that were easy to fit, so discarding the rest biases the interval narrow.
    """
    if problem.n_points - problem.n_slots < 1:
        # With no degrees of freedom the residuals are identically 0, so every resample returns the same answer.
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


def _corrected_aic(aic: float, n_points: int, n_params: int) -> float:
    """AICc: the Akaike criterion with the small-sample correction.

    Plain AIC is only asymptotically unbiased, and titrations are nowhere near that
    limit. A typical fit here has 6 to 15 points and 3 to 6 parameters, so n/k lands
    between 2 and 5, where AIC systematically favours the model with more parameters.
    The correction is recommended whenever n/k is below about 40.

    Returns infinity when the correction is undefined, which is when the sample is too
    small to support the parameter count at all.
    """
    if not np.isfinite(aic):
        return aic
    remaining = n_points - n_params - 1
    if remaining <= 0:
        return float("inf")
    return aic + 2.0 * n_params * (n_params + 1) / remaining


@dataclass(frozen=True)
class GlobalFitResult:
    """Result of a fit over several datasets.

    Attributes:
        model: The model that was fitted.
        params: `{dataset name: {parameter name: value}}`.
        intervals: `{dataset name: {parameter name: Interval}}`. Shared parameters
            carry the same interval in every dataset.
        shared: Names of the parameters that were shared.
        fixed: Names and values of the parameters that were held constant.
        method: Which interval method was used.
        r_squared: Coefficient of determination over all datasets combined.
        r_squared_per: Coefficient of determination per dataset.
        n_points: Total number of data points.
        n_points_per: Number of data points per dataset.
        n_free_params: Number of estimated parameters.
        aic: Akaike information criterion. Present for reference; prefer `aicc` for
            comparisons at these sample sizes.
        aicc: Akaike criterion with the small-sample correction. This is the one to
            compare models on, whether that is shared against unshared parameters or
            one functional form against another. Lower is better.
        unit: Name of the concentration unit, used for display only.
        fit_diagnostics: Findings that concern the entire fit, such as no degrees of
            freedom or bootstrap failure.
        diagnostics_per: Findings per dataset. Each `Diagnostic` has a stable code
            for programmatic handling and an English human-readable message.
        statistics_per: Raw statistic and p-value per dataset behind the
            residual-shape and heteroscedasticity checks, for applying your own
            multiple-comparison correction across datasets. See `Statistic`.
    """

    model: Model
    params: dict[str, dict[str, float]]
    intervals: dict[str, dict[str, Interval]]
    shared: tuple[str, ...]
    fixed: dict[str, float]
    method: Method
    r_squared: float
    r_squared_per: dict[str, float]
    n_points: int
    n_points_per: dict[str, int]
    n_free_params: int
    aic: float
    aicc: float
    unit: str = ""
    fit_diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)
    diagnostics_per: dict[str, tuple[Diagnostic, ...]] = field(default_factory=dict)
    statistics_per: dict[str, tuple[Statistic, ...]] = field(default_factory=dict)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.params)

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        """All warning diagnostics. Use `diagnostics_per` to retain dataset scope."""
        return tuple(
            diagnostic
            for diagnostics in (self.fit_diagnostics, *self.diagnostics_per.values())
            for diagnostic in diagnostics
            if diagnostic.severity == "warning"
        )

    @property
    def notes(self) -> tuple[Diagnostic, ...]:
        """All note diagnostics. Use `diagnostics_per` to retain dataset scope."""
        return tuple(
            diagnostic
            for diagnostics in (self.fit_diagnostics, *self.diagnostics_per.values())
            for diagnostic in diagnostics
            if diagnostic.severity == "note"
        )

    @property
    def warnings_per(self) -> dict[str, tuple[Diagnostic, ...]]:
        """Warning diagnostics per dataset."""
        return {
            name: tuple(diagnostic for diagnostic in diagnostics if diagnostic.severity == "warning")
            for name, diagnostics in self.diagnostics_per.items()
        }

    @property
    def notes_per(self) -> dict[str, tuple[Diagnostic, ...]]:
        """Note diagnostics per dataset."""
        return {
            name: tuple(diagnostic for diagnostic in diagnostics if diagnostic.severity == "note")
            for name, diagnostics in self.diagnostics_per.items()
        }

    def result_for(self, name: str) -> FitResult:
        """Extract one dataset as a `FitResult` with local and fit-wide diagnostics.

        Args:
            name: Dataset name.

        Returns:
            FitResult for that dataset.

        Raises:
            KeyError: If no dataset has that name.
        """
        if name not in self.params:
            raise KeyError(f"No such dataset: {name!r}. Available: {self.names}")
        return FitResult(
            model=self.model,
            params=dict(self.params[name]),
            intervals=dict(self.intervals[name]),
            r_squared=self.r_squared_per[name],
            n_points=self.n_points_per[name],
            fixed=dict(self.fixed),
            method=self.method,
            aic=self.aic,
            aicc=self.aicc,
            unit=self.unit,
            diagnostics=tuple(self.fit_diagnostics) + tuple(self.diagnostics_per.get(name, ())),
            statistics=self.statistics_per.get(name, ()),
        )

    def report(self) -> str:
        """Render the fit and structured diagnostics as human-readable text."""
        spec: list[str] = []
        if self.shared:
            spec.append("shared: " + ", ".join(self.shared))
        if self.fixed:
            spec.append("fixed: " + ", ".join(f"{key}={value:g}" for key, value in self.fixed.items()))
        lines = [
            f"model: {self.model.name}  ({self.model.description})",
            "global fit ("
            + ("; ".join(spec) if spec else "no shared or fixed parameters")
            + ")"
            + f"  {self.n_free_params} free parameters / {self.n_points} total points",
            f"interval: {self.method}",
            "",
        ]
        for name in self.names:
            lines.append(f"[{name}]  R^2 = {self.r_squared_per[name]:.4f}  (n = {self.n_points_per[name]})")
            width = max(len(self.model.label(param)) for param in self.model.params)
            for param in self.model.params:
                label = self.model.label(param).ljust(width)
                unit = self.unit if param == self.model.location else ""
                if param in self.fixed:
                    lines.append(f"  {label} = {self.params[name][param]:.4g}{' ' + unit if unit else ''}  (fixed)")
                else:
                    lines.append(f"  {label} = {self.intervals[name][param].format(unit)}")
            lines.append("")

        lines.append(f"overall R^2 = {self.r_squared:.4f}   AICc = {self.aicc:.2f}   (AIC = {self.aic:.2f})")
        lines.extend(
            f"{diagnostic.severity.upper()} [{diagnostic.code}]: {diagnostic.message}"
            for diagnostic in self.fit_diagnostics
        )
        for name in self.names:
            lines.extend(
                f"{diagnostic.severity.upper()} [{name}] [{diagnostic.code}]: {diagnostic.message}"
                for diagnostic in self.diagnostics_per.get(name, ())
            )
        if not self.fit_diagnostics and not any(self.diagnostics_per.values()):
            lines.append("No diagnostic issues detected.")
        return "\n".join(lines)


def _validate(
    datasets: Sequence[Dataset],
    model: Model,
    shared: Iterable[str],
    fixed: Mapping[str, float],
) -> tuple[tuple[str, ...], dict[str, float]]:
    if not datasets:
        raise ValueError("No datasets were given.")
    names = [d.name for d in datasets]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate dataset names: {names}")

    shared_t = tuple(dict.fromkeys(shared))
    fixed_d = dict(fixed)
    for group, label in ((shared_t, "shared"), (tuple(fixed_d), "fixed")):
        unknown = [p for p in group if p not in model.params]
        if unknown:
            raise ValueError(
                f"Unknown parameter name(s) in {label}: {unknown}. Model {model.name!r} has {list(model.params)}."
            )
    both = [p for p in shared_t if p in fixed_d]
    if both:
        raise ValueError(f"A parameter cannot be both shared and fixed: {both}")
    if len(fixed_d) == len(model.params):
        raise ValueError("Fixing every parameter leaves nothing to estimate.")

    # Sigma given for only some of the datasets is not allowed. A dataset without sigma has its weights implicitly
    # fixed at 1, so the ratio of weights between the two is decided by the absolute scale of sigma. Changing the unit
    # alone would move the result, so the promise that only the relative sizes matter would not hold. Statistically
    # too, error information for only some of the datasets has no settled interpretation.
    with_sigma = [d.name for d in datasets if d.sigma is not None]
    without_sigma = [d.name for d in datasets if d.sigma is None]
    if with_sigma and without_sigma:
        raise ValueError(
            f"sigma must be given for every dataset or for none. With sigma: {with_sigma}. Without: {without_sigma}."
        )
    return shared_t, fixed_d


def fit_global(
    datasets: Sequence[Dataset],
    model: Model = langmuir,
    shared: Iterable[str] = (),
    fixed: Mapping[str, float] | None = None,
    unit: str = "",
    ci: Method = "profile",
    n_boot: int = 1000,
    seed: int = 0,
) -> GlobalFitResult:
    """Fit several datasets simultaneously.

    Args:
        datasets: Sequence of `Dataset`. A single element gives an ordinary fit with
            some parameters held constant.
        model: The model to fit.
        shared: Parameter names to share across all datasets.
        fixed: Parameter names and the constants to hold them at. Applied to every
            dataset.
        unit: Name of the concentration unit, used for display only.
        ci: How to compute confidence intervals: "profile" (default), "asymptotic"
            or "bootstrap".
        n_boot: Number of bootstrap resamples, used when `ci="bootstrap"`. Must be at
            least `MIN_BOOTSTRAP_SAMPLES`.
        seed: Seed for the bootstrap resampling.

    Returns:
        GlobalFitResult

    Raises:
        ValueError: If the datasets or the shared/fixed specification are invalid, or
            `n_boot` is too small to form a percentile interval.
        RuntimeError: If the optimisation does not converge.

    Note:
        Residuals from the datasets are concatenated, so a dataset whose signal is
        larger dominates unless `Dataset.sigma` says otherwise. Give sigma, or
        normalise the signals onto a common scale, when they differ in magnitude.
    """
    shared_t, fixed_d = _validate(datasets, model, shared, fixed or {})
    if ci == "bootstrap" and n_boot < MIN_BOOTSTRAP_SAMPLES:
        raise ValueError(
            f"n_boot={n_boot} is below the {MIN_BOOTSTRAP_SAMPLES} resamples a percentile "
            "interval needs; the fit would return no interval at all. Raise n_boot or use "
            "ci='profile'."
        )
    problem = _Problem(datasets, model, shared_t, fixed_d)
    x, ssr, jac = problem.solve()

    bootstrap_failures = 0
    if ci == "asymptotic":
        if jac is None:  # pragma: no cover - a solve without pinned slots always returns a Jacobian
            raise RuntimeError("Jacobian unavailable; asymptotic intervals cannot be computed.")
        slot_intervals = _asymptotic_intervals(problem, x, ssr, jac)
    elif ci == "profile":
        slot_intervals = _profile_intervals(problem, x, ssr)
    elif ci == "bootstrap":
        slot_intervals, bootstrap_failures = _bootstrap_intervals(problem, x, n_boot, seed)
    else:
        raise ValueError(f"Unknown ci method: {ci!r}. Use 'asymptotic', 'profile' or 'bootstrap'.")

    params: dict[str, dict[str, float]] = {}
    intervals: dict[str, dict[str, Interval]] = {}
    r2_per: dict[str, float] = {}
    n_per: dict[str, int] = {}
    for i, d in enumerate(problem.datasets):
        values = problem.unpack(x, i)
        params[d.name] = dict(zip(model.params, values, strict=True))
        intervals[d.name] = {}
        for param in model.params:
            if param in fixed_d:
                held = float(fixed_d[param])
                intervals[d.name][param] = Interval(point=held, lower=held, upper=held, method=ci)
            else:
                intervals[d.name][param] = slot_intervals[problem.slot_of(param, i)]
        resid = d.observed - model(d.conc, *values)
        centered = d.observed - d.observed.mean()
        denom = float(centered @ centered)
        r2_per[d.name] = 1.0 - float(resid @ resid) / denom if denom > 0 else float("nan")
        n_per[d.name] = int(d.conc.size)

    # The coefficient of determination is descriptive, so it is left unweighted; ssr is weighted and cannot be reused.
    all_signal = np.concatenate([d.observed for d in problem.datasets])
    unweighted_ss_res = float(
        sum(
            float(resid @ resid)
            for resid in (d.observed - model(d.conc, *problem.unpack(x, i)) for i, d in enumerate(problem.datasets))
        )
    )
    centered_all = all_signal - all_signal.mean()
    ss_tot = float(centered_all @ centered_all)
    r_squared = 1.0 - unweighted_ss_res / ss_tot if ss_tot > 0 else float("nan")
    aic = problem.n_points * np.log(ssr / problem.n_points) + 2 * problem.n_slots if ssr > 0 else -np.inf
    aicc = _corrected_aic(float(aic), problem.n_points, problem.n_slots)

    fit_diagnostics: list[Diagnostic] = []
    per_diagnostics: dict[str, list[Diagnostic]] = {dataset.name: [] for dataset in problem.datasets}
    per_statistics: dict[str, list[Statistic]] = {dataset.name: [] for dataset in problem.datasets}

    # Degree-of-freedom and rank problems do not depend on the interval method, so they belong to the fit as a whole.
    dof = problem.n_points - problem.n_slots
    if dof < 1:
        fit_diagnostics.append(_diagnostic("no_degrees_of_freedom", "warning"))
    elif jac is not None:
        rank = _jacobian_rank(jac)
        if rank < problem.n_slots:
            fit_diagnostics.append(_diagnostic("rank_deficient_jacobian", "warning"))

    # The resamples that converge are the ones that were easy to fit, so the number of failures is reported.
    if bootstrap_failures:
        successes = n_boot - bootstrap_failures
        if successes < MIN_BOOTSTRAP_SAMPLES:
            fit_diagnostics.append(_diagnostic("bootstrap_insufficient_samples", "warning"))
        else:
            fit_diagnostics.append(_diagnostic("bootstrap_failures", "warning"))

    # Suppress the remarks whose premise changes under sharing or fixing, so the advice does not contradict itself.
    suppressed: set[str] = set()
    if model.amplitude in shared_t:
        suppressed |= {"not_saturated", "weakly_saturated"}
    if model.baseline is not None and model.baseline in fixed_d:
        suppressed.add("no_low_conc")

    for dataset in problem.datasets:
        loc = params[dataset.name][model.location]
        codes: set[str] = set()
        dataset_stats: list[Statistic] = []
        for code, _ in _diagnose_coded(
            dataset.conc,
            dataset.observed,
            model,
            params[dataset.name],
            intervals[dataset.name],
            dataset.receptor_conc,
            r2_per[dataset.name],
            tuple(fixed_d),
            dataset.sigma is not None,
            dataset_stats,
        ):
            if code not in suppressed:
                codes.add(code)
                per_diagnostics[dataset.name].append(_diagnostic(code, "warning"))
        per_statistics[dataset.name] = dataset_stats

        unsaturated = float(dataset.conc.max()) < 3 * loc
        # When the fit itself is broken, sharing cannot be credited with making the estimate possible; pairing that
        # with a warning that the location value is meaningless would be contradictory advice.
        broken = bool(codes & {"no_fit", "amplitude_collapsed"})
        if unsaturated and model.amplitude in shared_t and not broken:
            per_diagnostics[dataset.name].append(_diagnostic("shared_amplitude_identifies_location", "note"))
        elif unsaturated and model.amplitude not in shared_t and len(problem.datasets) > 1:
            per_diagnostics[dataset.name].append(_diagnostic("unshared_amplitude", "warning"))

    return GlobalFitResult(
        model=model,
        params=params,
        intervals=intervals,
        shared=shared_t,
        fixed=fixed_d,
        method=ci,
        r_squared=r_squared,
        r_squared_per=r2_per,
        n_points=problem.n_points,
        n_points_per=n_per,
        n_free_params=problem.n_slots,
        aic=float(aic),
        aicc=float(aicc),
        unit=unit,
        fit_diagnostics=tuple(fit_diagnostics),
        diagnostics_per={name: tuple(diagnostics) for name, diagnostics in per_diagnostics.items()},
        statistics_per={name: tuple(values) for name, values in per_statistics.items()},
    )


def fit(
    conc: NDArray[np.float64],
    signal: NDArray[np.float64],
    model: Model = langmuir,
    fixed: Mapping[str, float] | None = None,
    receptor_conc: float | None = None,
    unit: str = "",
    ci: Method = "profile",
    replicates: NDArray[np.float64] | None = None,
    sigma: NDArray[np.float64] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> FitResult:
    """Fit one dataset and return the parameters with 95% confidence intervals.

    Args:
        conc: Ligand concentration.
        signal: Observed signal, such as SPR response units, fluorescence
            intensity or luminescence.
        model: The model to fit.
        fixed: Parameter names and the constants to hold them at.
        receptor_conc: Concentration of the immobilised or fixed partner such as a
            receptor or lectin. Enables the ligand-depletion check when given.
        unit: Name of the concentration unit, used for display only.
        ci: How to compute confidence intervals: "profile" (default), "asymptotic"
            or "bootstrap".
        replicates: Array of shape `(n_replicates, n_points)` of repeat
            measurements, used when `ci="bootstrap"`.
        sigma: Per-point standard deviations. Give them when the measurement error
            is not the same size at every point, which is the case for fluorescence,
            luminescence and absorbance. Only the relative sizes matter.
        n_boot: Number of bootstrap resamples.
        seed: Seed for the bootstrap resampling.

    Returns:
        FitResult

    Raises:
        ValueError: If fewer data points than parameters are given.
        RuntimeError: If the optimisation does not converge.
    """
    conc = np.asarray(conc, dtype=float)
    signal = np.asarray(signal, dtype=float)
    fixed_d = dict(fixed or {})
    n_estimated = len(model.params) - len(fixed_d)
    if len(conc) < n_estimated:
        raise ValueError(f"Only {len(conc)} data point(s) for {n_estimated} estimated parameter(s).")

    name = "data"
    result = fit_global(
        [Dataset(name, conc, signal, receptor_conc, replicates, sigma)],
        model=model,
        fixed=fixed_d,
        unit=unit,
        ci=ci,
        n_boot=n_boot,
        seed=seed,
    )
    # Leave the extraction to result_for, which keeps the passing on of diagnostics in one place.
    return result.result_for(name)
