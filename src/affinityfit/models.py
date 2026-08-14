"""Model definitions.

A `Model` bundles what the fitting code needs about a functional form: parameter
names, the function, bounds, a starting guess, and which parameter plays which role.
Adding a model does not require touching the fitting or diagnostic code.

Every model here is a saturation curve, so each declares three roles:

- `location`: the concentration at half saturation. Kd for `langmuir`, Km for
  `michaelis`, IC50 for `ic50`.
- `amplitude`: the span of the observable between baseline and saturation.
- `baseline`: the signal at zero concentration, or None if the model has no offset.

Further roles are optional. `receptor` is declared by a model that solves ligand
depletion itself, and names the parameter holding the total concentration of the fixed
partner. `exponent` names the parameter that sets how steep the curve is, and
`cooperative` says whether that exponent is a claim about a mechanism or merely the
shape of the curve; the two are separate because a steepness worth reporting is not
always a cooperativity worth arguing over.

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
        receptor: Name of the parameter holding the total concentration of the fixed
            partner, or None when the model assumes it is dilute enough that binding
            does not measurably deplete the free ligand. Only a model that solves the
            depletion explicitly sets this; the diagnostics read it to decide whether
            recommending such a model still makes sense.
        exponent: Name of the parameter that decides how steep the curve is, or None
            when the model has none. Anything that reads the steepness looks here,
            whether to interpret it or to check that a formula relying on it being 1
            is allowed to be applied.
        cooperative: Whether `exponent` is a claim about binding mechanism rather than
            a description of the curve. True subscribes the model to the checks on
            whether cooperativity can be claimed; the slope of a dose-response curve
            is a shape parameter, so that model declares the exponent and leaves this
            False.
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
    receptor: str | None = None
    exponent: str | None = None
    cooperative: bool = False

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


def _tight_binding(
    conc: NDArray[np.float64], kd: float, bmax: float, baseline: float, rt: float
) -> NDArray[np.float64]:
    """Binding with ligand depletion solved exactly, written so that it keeps its precision.

    A hyperbola assumes the free ligand concentration equals the total one. That stops
    holding once the receptor is not much more dilute than Kd, because every molecule
    bound is one fewer left in solution. Solving the equilibrium without the assumption
    gives a quadratic in the complex concentration,

        [RL]^2 - (Rt + Lt + Kd) [RL] + Rt Lt = 0,

    whose physical root is the smaller one. Evaluated directly as
    `(b - sqrt(b^2 - 4 Rt Lt)) / 2` it subtracts two nearly equal numbers at low
    concentration and throws away most of the significant digits exactly where the
    curve is most informative. Multiplying by the conjugate turns that subtraction into
    an addition, and dividing through by `b` keeps `b^2` from overflowing:

        [RL] / Rt = 2 v / (1 + sqrt(1 - 4 u v)),   u = Rt / b,   v = Lt / b.

    Cancelling Rt is also what makes rt = 0 evaluable, where the expression collapses
    to the hyperbola `Lt / (Lt + Kd)` that `langmuir` describes.
    """
    total = rt + conc + kd
    # Every term is non-negative and Kd is bounded away from zero, so `total` is positive in any fit.
    # Substituting 1 covers a direct call with kd = 0 at zero concentration, where both numerators
    # are 0 as well and the fraction is 0 either way.
    scale = np.where(total > 0, total, 1.0)
    u = rt / scale
    v = conc / scale
    # 4 Rt Lt <= (Rt + Lt + Kd)^2 holds for any non-negative Kd, so the discriminant cannot be
    # negative; the clamp only absorbs rounding at the boundary, reached when Rt = Lt and Kd = 0.
    disc = np.maximum(1.0 - 4.0 * u * v, 0.0)
    return baseline + bmax * (2.0 * v / (1.0 + np.sqrt(disc)))


# ---------------------------------------------------------------- initial guesses


def _is_decreasing(conc: NDArray[np.float64], signal: NDArray[np.float64]) -> bool:
    """Whether the signal falls as the concentration rises.

    Compares the mean of the lower half of the concentrations with that of the upper
    half, which is robust to noise on individual points.
    """
    order = np.argsort(conc)
    half = max(1, len(conc) // 2)
    return bool(signal[order[-half:]].mean() < signal[order[:half]].mean())


def _guess_saturation(conc: NDArray[np.float64], signal: NDArray[np.float64]) -> dict[str, float]:
    baseline0 = float(signal[np.argmin(conc)])
    spread = float(signal.max() - signal.min())
    if spread <= 0:
        spread = float(abs(signal).max()) or 1.0

    # The sign of the amplitude is taken from the direction of the data. It is negative for an observable
    # that decreases on binding, such as fluorescence quenching.
    bmax0 = -spread if _is_decreasing(conc, signal) else spread

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


def _guess_ic50(conc: NDArray[np.float64], signal: NDArray[np.float64]) -> dict[str, float]:
    g = _guess_saturation(conc, signal)
    # A slope of 1 is the plain saturation curve, which is what a steeper or shallower one is judged
    # against, so it is where the search starts.
    return {"ic50": g["kd"], "bmax": g["bmax"], "baseline": g["baseline"], "hillslope": 1.0}


def _guess_tight_binding(conc: NDArray[np.float64], signal: NDArray[np.float64]) -> dict[str, float]:
    g = _guess_saturation(conc, signal)
    # Under depletion the observed midpoint sits near Kd + Rt/2, so it cannot be divided between the
    # two without knowing one of them. The receptor starts at a tenth of it, the point at which
    # depletion stops being negligible, which leaves the data free to pull it upwards. When rt is
    # supplied through `fixed=`, which is the usual case, this guess is never used.
    return {"kd": g["kd"], "bmax": g["bmax"], "baseline": g["baseline"], "rt": g["kd"] / 10.0}


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
    exponent="n",
    cooperative=True,
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

ic50 = Model(
    name="ic50",
    params=("ic50", "bmax", "baseline", "hillslope"),
    # The same algebra as `hill`. What differs is the vocabulary and, with it, the diagnostics: the
    # slope is declared as an exponent but not as a cooperative one, so the checks that ask whether
    # cooperativity can be claimed stay out of a dose-response report, where a slope near 1 is the
    # ordinary case rather than a finding. Declaring it at all is what lets a correction that assumes
    # a slope of 1, such as Cheng-Prusoff, find the slope and refuse to be applied blindly.
    func=_hill,
    bounds={"ic50": _POSITIVE, "bmax": _FREE, "baseline": _FREE, "hillslope": (0.05, 20.0)},
    initial=_guess_ic50,
    display={"ic50": "IC50", "bmax": "Bmax", "baseline": "baseline", "hillslope": "Hill slope"},
    location="ic50",
    amplitude="bmax",
    baseline="baseline",
    exponent="hillslope",
    # A positive Bmax turns the same curve into an activation (EC50) measurement; the half-maximal
    # concentration is still read from the `ic50` parameter.
    description="Dose-response (4PL): response = baseline + Bmax * L^h / (IC50^h + L^h); Bmax < 0 inhibits",
)

# The placement diagnostics compare the measured range against `kd`. Depletion makes the curve reach
# its plateau at a lower multiple of Kd than a hyperbola does, so those thresholds are conservative
# here rather than retuned: they can still ask for concentrations higher than this model needs.
tight_binding = Model(
    name="tight_binding",
    params=("kd", "bmax", "baseline", "rt"),
    func=_tight_binding,
    # Rt is left able to reach 0, which is the no-depletion limit. A fit that lands there is reported
    # by the stuck-at-a-bound check, and says that `langmuir` describes the data just as well.
    bounds={"kd": _POSITIVE, "bmax": _FREE, "baseline": _FREE, "rt": _NON_NEGATIVE},
    initial=_guess_tight_binding,
    display={"kd": "Kd", "bmax": "Bmax", "baseline": "baseline", "rt": "Rt"},
    location="kd",
    amplitude="bmax",
    baseline="baseline",
    receptor="rt",
    description="Tight binding: 1:1 binding solved for ligand depletion, Rt = total receptor",
)

MODELS: dict[str, Model] = {m.name: m for m in (langmuir, hill, michaelis, ic50, tight_binding)}
