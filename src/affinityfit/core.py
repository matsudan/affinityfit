"""Public fit diagnostics and compatibility exports.

The diagnostics assess fit quality and parameter identifiability using model roles
rather than literal parameter names. This module maps stable diagnostic codes to
user-facing messages and preserves the established imports for diagnostic records,
coded checks, and `FitResult`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from affinityfit.diagnostics import (
    Diagnostic,
    DiagnosticCode,
    Statistic,
    _diagnose_coded,
)
from affinityfit.diagnostics import (
    _at_bound as _at_bound,
)
from affinityfit.diagnostics import (
    _heteroscedastic as _heteroscedastic,
)
from affinityfit.diagnostics import (
    _no_fit as _no_fit,
)
from affinityfit.diagnostics import (
    _residual_structure as _residual_structure,
)
from affinityfit.models import Model
from affinityfit.results import FitResult as FitResult
from affinityfit.uncertainty import Interval

_DIAGNOSTIC_MESSAGES: dict[DiagnosticCode, str] = {
    DiagnosticCode.AMPLITUDE_COLLAPSED: (
        "The fitted amplitude has collapsed to nearly zero, so the fit is effectively flat."
    ),
    DiagnosticCode.NO_FIT: (
        "The fitted model is not distinguishable from a constant (F-test against the constant model), "
        "so the fitted location is not meaningful."
    ),
    DiagnosticCode.RESIDUAL_STRUCTURE: (
        "Residuals are systematically structured, so the model may not describe the mechanism."
    ),
    DiagnosticCode.HETEROSCEDASTIC: (
        "Residual variance grows with the fitted value; pass pointwise standard deviations with sigma=."
    ),
    DiagnosticCode.PARAM_AT_BOUND: (
        "A fitted parameter is pinned to a model bound and is set by the constraint rather than the data."
    ),
    DiagnosticCode.NOT_SATURATED: (
        "Saturation was not reached, so the location and amplitude cannot be identified separately."
    ),
    DiagnosticCode.WEAKLY_SATURATED: (
        "The measured range weakly constrains the plateau, so the location interval may be broad."
    ),
    DiagnosticCode.FEW_POINTS: "There are few data points relative to the number of estimated parameters.",
    DiagnosticCode.NO_POINTS_NEAR_KD: (
        "Fewer than two measurements lie near the fitted half-saturation concentration."
    ),
    DiagnosticCode.KD_EXTRAPOLATED: (
        "All measurements are on the saturated side, so the location is determined by extrapolation."
    ),
    DiagnosticCode.NO_LOW_CONC: "No sufficiently low-concentration measurement constrains the baseline.",
    DiagnosticCode.LIGAND_DEPLETION: (
        "Ligand depletion can bias the fitted location; use tight_binding for this experiment."
    ),
    DiagnosticCode.HILL_N_UNDETERMINED: (
        "The Hill coefficient interval cannot support a conclusion about cooperativity."
    ),
    DiagnosticCode.HILL_N_INCLUDES_ONE: (
        "The Hill coefficient interval includes one, so cooperativity cannot be claimed."
    ),
    DiagnosticCode.HILL_N_BELOW_ONE: (
        "The Hill coefficient is significantly below one and may indicate negative cooperativity or heterogeneity."
    ),
    DiagnosticCode.HILL_N_ABOVE_ONE: (
        "The Hill coefficient is significantly above one, but depletion, self-association, or pre-equilibrium "
        "readout can produce the same shape."
    ),
    DiagnosticCode.LIMIT_UNDETERMINED: (
        "At least one confidence-interval limit is undetermined; report a one-sided limit instead of the "
        "point estimate."
    ),
    DiagnosticCode.NO_DEGREES_OF_FREEDOM: (
        "There are no degrees of freedom, so no confidence interval can be calculated."
    ),
    DiagnosticCode.RANK_DEFICIENT_JACOBIAN: (
        "The Jacobian is rank-deficient, so the data cannot identify all parameter combinations."
    ),
    DiagnosticCode.BOOTSTRAP_INSUFFICIENT_SAMPLES: (
        "Too few bootstrap resamples converged to form a percentile interval."
    ),
    DiagnosticCode.BOOTSTRAP_FAILURES: (
        "Some bootstrap resamples did not converge, so the interval may be too narrow."
    ),
    DiagnosticCode.SHARED_AMPLITUDE_IDENTIFIES_LOCATION: (
        "Sharing the amplitude makes this otherwise unsaturated dataset identifiable."
    ),
    DiagnosticCode.UNSHARED_AMPLITUDE: (
        "This unsaturated dataset has a free amplitude; share it only when the maximum signal is justified as common."
    ),
}


def _diagnostic(code: DiagnosticCode, severity: Literal["warning", "note"]) -> Diagnostic:
    return Diagnostic(code=code, severity=severity, message=_DIAGNOSTIC_MESSAGES[code])


def diagnose(
    conc: NDArray[np.float64],
    signal: NDArray[np.float64],
    model: Model,
    params: dict[str, float],
    intervals: dict[str, Interval] | None = None,
    receptor_conc: float | None = None,
    fixed_names: tuple[str, ...] = (),
    weighted: bool = False,
    stats_out: list[Statistic] | None = None,
) -> tuple[Diagnostic, ...]:
    """Judge whether the estimated parameters can be trusted.

    Two families of problems are covered. The first is the health of the fit itself:
    a collapsed amplitude, a model that is not distinguishable from a constant,
    residuals that are systematically arranged rather than scattered, and parameters
    stuck at the edge of their allowed range. The second is the placement of the
    measurements: saturation never reached, too few points, no points near or below
    the half-saturation constant, ligand depletion, and a Hill coefficient whose
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
        fixed_names: Parameters that were held constant, exempt from the
            stuck-at-a-bound check and counted as already spent when judging
            whether the model explains the data better than its own mean would.
        weighted: Whether per-point sigma was supplied.
        stats_out: When given, the statistic and p-value behind the model-vs-constant,
            residual-shape and heteroscedasticity checks are appended to it. Pass a
            list to collect them for your own multiple-comparison correction across
            several fits; see `Statistic` and `FitResult.statistics`.

    Returns:
        A tuple of machine-readable diagnostics, empty when nothing was detected.
    """
    return tuple(
        _diagnostic(code, "warning")
        for code in _diagnose_coded(
            conc, signal, model, params, intervals, receptor_conc, fixed_names, weighted, stats_out
        )
    )
