"""Immutable fitted-result containers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from affinityfit.diagnostics import Diagnostic, Statistic
from affinityfit.models import Model
from affinityfit.uncertainty import Interval, Method


def _freeze_mapping[K, V](mapping: Mapping[K, V]) -> Mapping[K, V]:
    return MappingProxyType(dict(mapping))


def _freeze_nested_mapping[K, InnerK, V](
    mapping: Mapping[K, Mapping[InnerK, V]],
) -> Mapping[K, Mapping[InnerK, V]]:
    return _freeze_mapping({key: _freeze_mapping(inner) for key, inner in mapping.items()})


@dataclass(frozen=True)
class FitResult:
    """Fitted parameters together with diagnostic messages.

    Attributes:
        model: The model that was fitted.
        params: Read-only mapping of parameter names to values.
        intervals: Read-only mapping of parameter names to `Interval` objects.
            Intervals may be asymmetric, and one side may be None when that limit
            could not be determined.
        r_squared: Coefficient of determination.
        n_points: Number of data points.
        fixed: Read-only mapping of names to values for parameters that were held
            constant.
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
        receptor_conc: Total concentration of the fixed partner, as supplied to the
            fit. Retained because the corrections applied after a fit need it:
            `ki_from_ic50` uses it to replace the Cheng-Prusoff approximation with the
            exact form. None when it was not given.
    """

    model: Model
    params: Mapping[str, float]
    intervals: Mapping[str, Interval]
    r_squared: float
    n_points: int
    fixed: Mapping[str, float] = field(default_factory=dict)
    method: str = "profile"
    aic: float = float("nan")
    aicc: float = float("nan")
    unit: str = ""
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)
    statistics: tuple[Statistic, ...] = field(default_factory=tuple)
    receptor_conc: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", _freeze_mapping(self.params))
        object.__setattr__(self, "intervals", _freeze_mapping(self.intervals))
        object.__setattr__(self, "fixed", _freeze_mapping(self.fixed))

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
        if np.isfinite(self.aicc):
            lines.append(f"{'AICc'.ljust(width)} = {self.aicc:.2f}   (AIC = {self.aic:.2f})")
        lines.append(f"{'R^2'.ljust(width)} = {self.r_squared:.4f}   (n = {self.n_points}; descriptive only)")
        lines.append("")
        lines.extend(
            f"{diagnostic.severity.upper()} [{diagnostic.code}]: {diagnostic.message}"
            for diagnostic in self.diagnostics
        )
        if not self.diagnostics:
            lines.append("No diagnostic issues detected.")
        return "\n".join(lines)


@dataclass(frozen=True)
class GlobalFitResult:
    """Result of a fit over several datasets.

    All mapping attributes are copied during construction and are read-only.

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
        receptor_conc_per: Total concentration of the fixed partner per dataset, as
            supplied on each `Dataset`. Passed on by `result_for` so that corrections
            applied after the fit, such as `ki_from_ic50`, can use it.
    """

    model: Model
    params: Mapping[str, Mapping[str, float]]
    intervals: Mapping[str, Mapping[str, Interval]]
    shared: tuple[str, ...]
    fixed: Mapping[str, float]
    method: Method
    r_squared: float
    r_squared_per: Mapping[str, float]
    n_points: int
    n_points_per: Mapping[str, int]
    n_free_params: int
    aic: float
    aicc: float
    unit: str = ""
    fit_diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)
    diagnostics_per: Mapping[str, tuple[Diagnostic, ...]] = field(default_factory=dict)
    statistics_per: Mapping[str, tuple[Statistic, ...]] = field(default_factory=dict)
    receptor_conc_per: Mapping[str, float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", _freeze_nested_mapping(self.params))
        object.__setattr__(self, "intervals", _freeze_nested_mapping(self.intervals))
        object.__setattr__(self, "fixed", _freeze_mapping(self.fixed))
        object.__setattr__(self, "r_squared_per", _freeze_mapping(self.r_squared_per))
        object.__setattr__(self, "n_points_per", _freeze_mapping(self.n_points_per))
        object.__setattr__(self, "diagnostics_per", _freeze_mapping(self.diagnostics_per))
        object.__setattr__(self, "statistics_per", _freeze_mapping(self.statistics_per))
        object.__setattr__(self, "receptor_conc_per", _freeze_mapping(self.receptor_conc_per))

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

    def _diagnostics_of_severity_per(
        self,
        severity: Literal["warning", "note"],
    ) -> dict[str, tuple[Diagnostic, ...]]:
        fit_diagnostics = tuple(diagnostic for diagnostic in self.fit_diagnostics if diagnostic.severity == severity)
        return {
            name: fit_diagnostics
            + tuple(diagnostic for diagnostic in self.diagnostics_per.get(name, ()) if diagnostic.severity == severity)
            for name in self.names
        }

    @property
    def warnings_per(self) -> dict[str, tuple[Diagnostic, ...]]:
        """Warning diagnostics per dataset, including fit-wide diagnostics."""
        return self._diagnostics_of_severity_per("warning")

    @property
    def notes_per(self) -> dict[str, tuple[Diagnostic, ...]]:
        """Note diagnostics per dataset, including fit-wide diagnostics."""
        return self._diagnostics_of_severity_per("note")

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
            receptor_conc=self.receptor_conc_per.get(name),
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
            lines.append(
                f"[{name}]  n = {self.n_points_per[name]}, R^2 = {self.r_squared_per[name]:.4f} (descriptive only)"
            )
            width = max(len(self.model.label(param)) for param in self.model.params)
            for param in self.model.params:
                label = self.model.label(param).ljust(width)
                unit = self.unit if param == self.model.location else ""
                if param in self.fixed:
                    lines.append(f"  {label} = {self.params[name][param]:.4g}{' ' + unit if unit else ''}  (fixed)")
                else:
                    lines.append(f"  {label} = {self.intervals[name][param].format(unit)}")
            lines.append("")

        lines.append(
            f"overall AICc = {self.aicc:.2f}   (AIC = {self.aic:.2f})   R^2 = {self.r_squared:.4f} (descriptive only)"
        )
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
