"""Model definitions.

A `Model` bundles what the fitting code needs about a functional form: parameter
names, the function, bounds, a starting guess, and which parameter plays which role.
Adding a model does not require touching the fitting or diagnostic code.

Every model here is a saturation curve, so each declares three roles:

- `location`: the concentration at half saturation. Kd for `langmuir`, Km for
  `michaelis`.
- `amplitude`: the span of the observable between baseline and saturation.
- `baseline`: the signal at zero concentration, or None if the model has no offset.

The diagnostics are written against these roles rather than literal parameter names,
which is what lets one set of checks serve every model.

Models are callable, so the same object works as a plain function and as the
`model=` argument of a fit:

    langmuir(conc, 10.0, 1.0, 0.0)
    fit(conc, signal, model=langmuir)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Guess = Callable[[NDArray[np.float64], NDArray[np.float64]], dict[str, float]]


@dataclass(frozen=True)
class Model:
    """A functional form together with the metadata needed to fit it.

    Attributes:
        name: Short identifier, also used by the CLI.
        params: Parameter names in the order the function takes them.
        func: The function itself, called as `func(conc, *params)`.
        bounds: Lower and upper bound per parameter.
        initial: Callable returning a starting guess as `{name: value}`.
        display: Human-readable symbol per parameter, used in reports.
        location: Name of the parameter that is the half-saturation concentration.
        amplitude: Name of the parameter that is the signal span.
        baseline: Name of the offset parameter, or None if the model has none.
        description: One-line summary shown by the CLI.
    """

    name: str
    params: tuple[str, ...]
    func: Callable[..., NDArray[np.float64]]
    bounds: Mapping[str, tuple[float, float]]
    initial: Guess
    display: Mapping[str, str]
    location: str
    amplitude: str
    baseline: str | None
    description: str

    def __call__(self, conc: NDArray[np.float64] | float, *params: float) -> NDArray[np.float64]:
        """Evaluate the model, so it can be used as a plain function."""
        if len(params) != len(self.params):
            raise TypeError(f"{self.name} takes {len(self.params)} parameters {list(self.params)}, got {len(params)}")
        return self.func(np.asarray(conc, dtype=float), *params)

    def label(self, param: str) -> str:
        """Human-readable symbol for a parameter."""
        return self.display.get(param, param)

    def lower(self, param: str) -> float:
        return self.bounds[param][0]

    def upper(self, param: str) -> float:
        return self.bounds[param][1]

    def ordered(self, values: Mapping[str, float]) -> tuple[float, ...]:
        """Turn a `{name: value}` mapping into the positional order of `func`."""
        return tuple(float(values[p]) for p in self.params)

    def is_log_scale(self, param: str) -> bool:
        """Whether the optimiser should work with the base-10 logarithm of a parameter.

        True for parameters that are strictly positive and unbounded above, which is
        the shape of a concentration constant such as Kd or Km. Optimising such a
        parameter linearly makes the result depend on the unit chosen; working in the
        logarithm makes it scale-invariant, so the same affinity is recovered whether
        it is written as 1e-11 M, 1e-2 nM or 10 pM.

        Parameters with a finite upper bound (the Hill coefficient) or that may be
        negative (the amplitude of a quenching signal, the baseline) stay linear.
        """
        return self.lower(param) > 0.0 and not np.isfinite(self.upper(param))


# --------------------------------------------------------------------- functions


def _saturation(conc: NDArray[np.float64], kd: float, bmax: float, baseline: float) -> NDArray[np.float64]:
    return baseline + bmax * conc / (kd + conc)


def _hill(conc: NDArray[np.float64], kd: float, bmax: float, baseline: float, n: float) -> NDArray[np.float64]:
    """Cooperative binding, written so that it never returns NaN.

    The direct form `bmax * conc**n / (kd**n + conc**n)` divides zero by zero once
    `kd**n` underflows, which happens inside the declared bounds. Dividing through by
    `conc**n` gives `bmax / (1 + (kd/conc)**n)`, where an overflowing ratio saturates
    to infinity and the term goes to zero.
    """
    ratio = np.divide(kd, conc, out=np.full(np.shape(conc), np.inf, dtype=float), where=conc > 0)
    with np.errstate(over="ignore"):
        powered = np.power(ratio, n)
    return baseline + bmax / (1.0 + powered)


# ---------------------------------------------------------------- initial guesses


def _guess_saturation(conc: NDArray[np.float64], signal: NDArray[np.float64]) -> dict[str, float]:
    baseline0 = float(signal[np.argmin(conc)])
    spread = float(signal.max() - signal.min())
    if spread <= 0:
        spread = float(abs(signal).max()) or 1.0

    # The sign of the amplitude is taken from the direction of the data. It is negative for an observable
    # that decreases on binding, such as fluorescence quenching.
    order = np.argsort(conc)
    half = max(1, len(conc) // 2)
    low = float(signal[order[:half]].mean())
    high = float(signal[order[-half:]].mean())
    bmax0 = spread if high >= low else -spread

    target = baseline0 + bmax0 / 2.0
    kd0 = float(conc[np.argmin(np.abs(signal - target))])
    if kd0 <= 0:
        positive = conc[conc > 0]
        kd0 = float(np.median(positive)) if positive.size else 1.0
    return {"kd": kd0, "bmax": bmax0, "baseline": baseline0}


def _guess_hill(conc: NDArray[np.float64], signal: NDArray[np.float64]) -> dict[str, float]:
    # Start from the non-cooperative case; n is the parameter under test.
    return {**_guess_saturation(conc, signal), "n": 1.0}


def _guess_michaelis(conc: NDArray[np.float64], signal: NDArray[np.float64]) -> dict[str, float]:
    g = _guess_saturation(conc, signal)
    return {"km": g["kd"], "vmax": g["bmax"], "baseline": g["baseline"]}


# ----------------------------------------------------------------------- models

# Lower bound for the half-saturation constant. Because it is optimised on a logarithmic scale, the bound is
# only a marker for a physically impossible value and does not affect precision.
_POSITIVE = (1e-30, np.inf)
_NON_NEGATIVE = (0.0, np.inf)
_FREE = (-np.inf, np.inf)

langmuir = Model(
    name="langmuir",
    params=("kd", "bmax", "baseline"),
    func=_saturation,
    # The sign of bmax is left unconstrained. Kd means the same thing for a decreasing observable, which the
    # same expression describes with a negative response coefficient.
    bounds={"kd": _POSITIVE, "bmax": _FREE, "baseline": _FREE},
    initial=_guess_saturation,
    display={"kd": "Kd", "bmax": "Bmax", "baseline": "baseline"},
    location="kd",
    amplitude="bmax",
    baseline="baseline",
    description="1:1 binding: signal = baseline + Bmax * L / (Kd + L)",
)

hill = Model(
    name="hill",
    params=("kd", "bmax", "baseline", "n"),
    func=_hill,
    # n is kept away from 0 and from absurdly steep values; real cooperativity in
    # binding data sits well inside this range.
    bounds={"kd": _POSITIVE, "bmax": _FREE, "baseline": _FREE, "n": (0.05, 20.0)},
    initial=_guess_hill,
    display={"kd": "Kd(app)", "bmax": "Bmax", "baseline": "baseline", "n": "n (Hill)"},
    location="kd",
    amplitude="bmax",
    baseline="baseline",
    description="Cooperative binding: signal = baseline + Bmax * L^n / (Kd^n + L^n)",
)

michaelis = Model(
    name="michaelis",
    params=("km", "vmax", "baseline"),
    func=_saturation,
    # Vmax is a reaction rate, so it is kept non-negative. Use langmuir or hill for a decreasing observable.
    bounds={"km": _POSITIVE, "vmax": _NON_NEGATIVE, "baseline": _FREE},
    initial=_guess_michaelis,
    display={"km": "Km", "vmax": "Vmax", "baseline": "baseline"},
    location="km",
    amplitude="vmax",
    baseline="baseline",
    # Km equals (k_off + k_cat) / k_on and only coincides with Kd when k_cat is
    # negligible, so it is not an affinity in general even though the algebra is
    # identical to `langmuir`.
    description="Michaelis-Menten: v = baseline + Vmax * S / (Km + S)",
)

MODELS: dict[str, Model] = {m.name: m for m in (langmuir, hill, michaelis)}
