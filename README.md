# affinityfit

[![PyPI](https://img.shields.io/pypi/v/affinityfit)](https://pypi.org/project/affinityfit/)
[![Python versions](https://img.shields.io/pypi/pyversions/affinityfit)](https://pypi.org/project/affinityfit/)
[![ci](https://github.com/matsudan/affinityfit/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/matsudan/affinityfit/actions/workflows/ci.yml)

🚧 This library is currently under development and may change substantially.

A Python library that fits *K*<sub>d</sub> and related parameters from concentration and
signal data, and diagnoses whether the measurements constrain the resulting estimate.

Any observable that is linear in the fraction bound can be fitted, such as nuclear
magnetic resonance (NMR) peak intensity, surface plasmon resonance (SPR) steady-state
response, or initial enzyme velocity. Several datasets can be fitted simultaneously,
sharing parameters across them or holding some constant.

```
signal = baseline + Bmax * [L] / (Kd + [L])
```

## Usage

```bash
uv add affinityfit
```

```python
import numpy as np
from affinityfit import DiagnosticCode, fit

conc, signal = np.loadtxt(
    "titration.csv",
    delimiter=",",
    skiprows=1,
    unpack=True,
)
res = fit(conc, signal, unit="nM")

print(res.params["kd"], res.intervals["kd"].format("nM"))

# `code` is the stable programmatic contract. `severity` classifies the finding.
for diagnostic in res.diagnostics:
    print(diagnostic.code, diagnostic.severity, diagnostic.message)

if any(diagnostic.code == DiagnosticCode.NOT_SATURATED for diagnostic in res.diagnostics):
    # Extend the concentration range before interpreting Kd or Bmax.
    pass

# Optional, caller-controlled text rendering for a terminal or notebook.
print(res.report())
```

Each `Diagnostic` in `res.diagnostics` has a stable `code` to branch on, a `severity`
of `"warning"` or `"note"`, and an English `message` for display. Do not branch on
`message`, which may be reworded. `res.warnings` and `res.notes` filter by severity.
`list(DiagnosticCode)` enumerates every code. Members are plain strings, so
`diagnostic.code == "not_saturated"` works as well.

The CSV has one header row, with concentration in the first column and signal in the
second.

```csv
concentration_nM,signal
0,0.0249
1,0.0963
3,0.2493
10,0.5090
30,0.7657
100,0.9249
300,0.9758
1000,1.0035
```

Output of `report()`:

```
model    : langmuir  (1:1 binding: signal = baseline + Bmax * L / (Kd + L))
interval : profile
Kd       = 10.1 +/- 0.8 nM
Bmax     = 0.995 +/- 0.017
baseline = 0.018 +/- 0.014
AICc     = -63.79   (AIC = -77.12)
R^2      = 0.9998   (n = 8; descriptive only)

No diagnostic issues detected.
```

If the concentration of the fixed partner (for example, a receptor or a lectin) is
known, pass it in to enable the ligand-depletion check.

```python
res = fit(conc, signal, receptor_conc=1.0, unit="nM")
```

Pass `model=` to switch models.

```python
from affinityfit import hill

res = fit(conc, signal, model=hill)  # test for cooperativity
print(res.intervals["n"].contains(1.0))
```

Fitting several datasets simultaneously, sharing or fixing parameters:

```python
from affinityfit import Dataset, fit_global

res = fit_global(
    [Dataset("oxidized", conc, sig_ox), Dataset("reduced", conc, sig_red)],
    shared=["bmax"],  # estimate a single value across all datasets
    fixed={"baseline": 0.0},  # hold at a constant, not estimated
    unit="mM",
)
# Preserve diagnostic scope in a global fit.
for diagnostic in res.fit_diagnostics:
    print("fit", diagnostic.code, diagnostic.severity)
for name, diagnostics in res.diagnostics_per.items():
    for diagnostic in diagnostics:
        print(name, diagnostic.code, diagnostic.severity)

# Pull out one dataset. Its local and fit-wide diagnostics come along with it.
sub = res.result_for("oxidized")
x, y = sub.curve()

# Optional caller-controlled rendering.
print(res.report())
```

`fit_diagnostics` holds findings about the whole fit, and `diagnostics_per` those of
each dataset. `result_for()` returns one dataset's `FitResult` with both. `warnings` and
`notes` span both scopes; `warnings_per` and `notes_per` give the per-dataset view,
including fit-wide findings.

## Weighting

By default every point counts equally, which assumes the measurement error is the same
size everywhere. For fluorescence, luminescence, or absorbance, where the error grows
with the signal, pass the per-point standard deviation as `sigma`. Without it the
confidence interval comes out too narrow.

```python
res = fit(conc, signal, sigma=0.01 + 0.10 * signal)
```

Only the ratios between `sigma` values matter: the overall scale is estimated from the
residuals, so multiplying `sigma` by a constant does not change the result. In a global
fit, `Dataset(..., sigma=...)` also sets the relative weight between datasets, and must
be given for every dataset or for none; mixing the two raises an error.

`HETEROSCEDASTIC` warns when the residuals grow with the fitted value.

## Confidence intervals

`ci=` selects one of three methods.

| Method | Description |
|---|---|
| `profile` (default) | Pins each parameter in turn, refits the others, and takes the bounds from an F-test on the residual sum of squares. The interval can be asymmetric; an undetermined side is `None` and formatted as a one-sided limit |
| `asymptotic` | Reads the interval from the covariance matrix. Fastest, but undetermined when there is no residual degree of freedom or the Jacobian is rank-deficient |
| `bootstrap` | Resamples replicates when given, residuals otherwise, and refits |

`replicates` has shape `(n_replicates, n_points)`.

```python
res = fit(conc, signal, ci="bootstrap", replicates=reps, n_boot=2000)
```

`Dataset` also accepts replicates on their own (`signal` is then their mean).

```python
Dataset("oxidized", conc, replicates=reps)
```

Under `ci="bootstrap"`, `n_boot` below 100 raises an error.

Values are rounded to the precision their uncertainty supports, for example
`(3.8 +/- 0.6)e-08`.

## Model

| Model | Parameters | Use |
|---|---|---|
| `langmuir` (default) | kd, bmax, baseline | 1:1 binding. `signal = baseline + Bmax·L/(Kd+L)` |
| `hill` | kd, bmax, baseline, n | Cooperativity, judged by whether the confidence interval of *n* contains 1. See [Ligand depletion](#ligand-depletion) before claiming *n* > 1 |
| `michaelis` | km, vmax, baseline | Enzyme kinetics. Same equation as langmuir, but *K*<sub>m</sub> is not an affinity |
| `ic50` | ic50, bmax, baseline, hillslope | Dose–response (four-parameter logistic, 4PL). A negative `bmax` inhibits, a positive one gives an EC<sub>50</sub> curve |
| `tight_binding` | kd, bmax, baseline, rt | 1:1 binding solved for ligand depletion, for a receptor not dilute against *K*<sub>d</sub> |

`bmax` in `langmuir` and `hill` may be negative, for a signal that decreases on binding.
`vmax` in `michaelis` stays non-negative, so use `langmuir` or `hill` for decreasing
data.

*K*<sub>d</sub>, *K*<sub>m</sub>, and IC<sub>50</sub> are optimised on a logarithmic
scale, so the result does not depend on the concentration unit.

### Dose–response and *K*<sub>i</sub>

`ic50` is the same equation as `hill`, without the cooperativity checks.

```python
from affinityfit import ic50

res = fit(conc, response, model=ic50, unit="nM")
print(res.params["ic50"], res.params["hillslope"])
```

`ki_from_ic50` converts a displacement IC<sub>50</sub> into *K*<sub>i</sub>.

```python
from affinityfit import Interval, ki_from_ic50

ki = ki_from_ic50(
    res,
    tracer_conc=5.0,
    tracer_kd=Interval(point=2.0, lower=1.6, upper=2.4),
    receptor_conc=3.45,  # selects the exact correction
)
print(ki.format("nM"))
```

- **Pass the whole `FitResult`**, not `res.intervals["ic50"]`. Only the result carries
  the slope for the check below and the `receptor_conc` given to `fit()`.
- **Give `receptor_conc` whenever it is known.** It selects the exact correction.
  Without it the Cheng–Prusoff form `IC50 / (1 + [T]/Kd)` is used, which is biased when
  the receptor depletes the tracer, and a `UserWarning` says so.
- **Give `tracer_kd` as an `Interval`** when its uncertainty is known; it is propagated
  into *K*<sub>i</sub> and can dominate it. `tracer_conc` and `receptor_conc` are taken
  as exact.
- **The correction assumes a slope of 1.** A `UserWarning` is raised when the fitted
  slope's interval excludes 1. Ligand depletion steepens the curve, so rule it out first.

For competitive enzyme inhibition, pass `[S]` and `Km` in place of the tracer.

### Ligand depletion

Every model except `tight_binding` assumes the free ligand concentration equals the total
one. This fails once the receptor is not much more dilute than *K*<sub>d</sub>:
*K*<sub>d</sub> is overestimated while *R*<sup>2</sup> stays high. `tight_binding`
solves the 1:1 equilibrium without that assumption. Pass the total receptor
concentration as `rt`.

```python
from affinityfit import tight_binding

res = fit(conc, signal, model=tight_binding, fixed={"rt": 5.0}, unit="uM")
```

When `rt` is left free, it is estimated as the active receptor concentration.

```python
res = fit(conc, signal, model=tight_binding, unit="uM")
print(res.intervals["rt"].format("uM"))
```

- `fixed=` applies one value to every dataset, so fit datasets at different receptor
  concentrations separately.
- The saturation checks may ask for higher concentrations than this model requires.

Depletion also steepens the curve, so `hill` reports *n* > 1 without any cooperativity.
Measure cooperativity with the receptor at or below *K*<sub>d</sub>/10.

## Model selection

Compare models or parameter-sharing schemes by `aicc`; lower is preferred. `aic` is kept
for reference, but favours the model with more parameters at typical titration sizes.

```python
from affinityfit import fit_global, hill, langmuir

free = fit_global(datasets, model=langmuir)
shared = fit_global(datasets, model=langmuir, shared=["bmax"])
cooperative = fit_global(datasets, model=hill)
print(free.aicc, shared.aicc, cooperative.aicc)
```

`k` counts the residual variance as a parameter. When *n* − *k* − 1 ≤ 0, `aicc` is
infinite and `report()` omits the line.

## Diagnostics

A high *R*<sup>2</sup> does not mean *K*<sub>d</sub> is determined. The conditions below
are detected automatically; branch on `Diagnostic.code`.

| Condition | Meaning | Code |
|---|---|---|
| Amplitude within 1% of the signal range | The fit is effectively a flat line; the model cannot express the shape of the data | `AMPLITUDE_COLLAPSED` |
| Fitted model not distinguishable from its own mean (F-test against a constant, *P* ≥ 0.01) | The model does not capture the trend in the data; the value of *K*<sub>d</sub> is meaningless | `NO_FIT` |
| Systematic sign in the residuals | The shape of the model does not match the mechanism, even with a high coefficient of determination. Needs at least eight points | `RESIDUAL_STRUCTURE` |
| A parameter stuck at a bound | That value is an artefact of the constraint, not an estimate, and cannot be reported | `PARAM_AT_BOUND` |
| Highest concentration < 3 × *K*<sub>d</sub> | Saturation was not reached; the fitted location and amplitude may not be identifiable separately. Extend the measured range before interpreting either value | `NOT_SATURATED` |
| Highest concentration < 10 × *K*<sub>d</sub> | The estimate of *B*<sub>max</sub> is unstable, and the confidence interval on *K*<sub>d</sub> widens as well | `WEAKLY_SATURATED` |
| Data points < 2 × estimated parameters | Not enough information; confidence intervals are indicative only (fixed parameters are not counted) | `FEW_POINTS` |
| Fewer than two points near *K*<sub>d</sub> (*K*<sub>d</sub>/3 to 3 × *K*<sub>d</sub>) | The inflection point of the curve is underdetermined; adding points here improves the estimate the most | `NO_POINTS_NEAR_KD` |
| Lowest concentration > *K*<sub>d</sub> | Every point sits on the saturated side; *K*<sub>d</sub> is set by extrapolation and should not be reported to extra significant figures | `KD_EXTRAPOLATED` |
| No points at or below *K*<sub>d</sub>/10 | baseline is estimated together with the curve, which can shift *B*<sub>max</sub> | `NO_LOW_CONC` |
| Receptor concentration > *K*<sub>d</sub>/10 | Ligand depletion causes *K*<sub>d</sub> to be overestimated; switch to `tight_binding`. Not reported when that model is already in use. With `hill`, also states that *n* is inflated and cooperativity cannot be judged at all under these conditions | `LIGAND_DEPLETION` |
| The confidence interval of the Hill coefficient *n* is undetermined | The residuals leave no scatter from which to determine a direction, or a side is undetermined; cooperativity cannot be judged | `HILL_N_UNDETERMINED` |
| The confidence interval of the Hill coefficient *n* contains 1 | Cooperativity cannot be claimed | `HILL_N_INCLUDES_ONE` |
| Hill coefficient *n* significantly > 1 | Positive cooperativity is one reading, but depletion, self-association, and a pre-equilibrium reading give the same shape | `HILL_N_ABOVE_ONE` |
| Hill coefficient *n* significantly < 1 | Negative cooperativity, heterogeneous sites, or a heterogeneous sample | `HILL_N_BELOW_ONE` |
| Positive rank association between absolute residuals and fitted values (one-sided Spearman test, *P* < 0.01) | Heteroscedastic error; omitting `sigma` narrows the interval. Needs at least eight points, and is not checked once `sigma` is supplied | `HETEROSCEDASTIC` |
| Zero degrees of freedom (points ≤ parameters) | No residual variance is available to estimate uncertainty; no confidence interval can be calculated | `NO_DEGREES_OF_FREEDOM` |
| Rank-deficient Jacobian | Parameters cannot be distinguished, so the values are not uniquely determined; more concentrations are needed | `RANK_DEFICIENT_JACOBIAN` |
| One side of a confidence interval is undetermined | The point estimate should not be reported; sharing or a wider measured range is needed. The Hill coefficient is excluded, the `HILL_N_*` codes covering it instead | `LIMIT_UNDETERMINED` |
| Too many bootstrap resamples failed to converge | Below the minimum for a percentile interval; the interval is reported as undetermined | `BOOTSTRAP_INSUFFICIENT_SAMPLES` |
| Some bootstrap resamples failed to converge | The interval may be narrower than it should be, since the resamples that converge are the easier ones to fit | `BOOTSTRAP_FAILURES` |
| Amplitude shared in a global fit rescues an unsaturated dataset | Sharing rendered an otherwise unidentifiable estimate identifiable; not a problem | `SHARED_AMPLITUDE_IDENTIFIES_LOCATION` |
| Amplitude free in a global fit with an unsaturated dataset | Consider sharing the amplitude if the maximum signal is common across datasets | `UNSHARED_AMPLITUDE` |

- In a global fit, sharing the amplitude suppresses `NOT_SATURATED` and
  `WEAKLY_SATURATED`, and fixing the baseline suppresses `NO_LOW_CONC`.
- When a confidence interval cannot be computed, it is reported as undetermined with the
  reason; no value is substituted. The point estimate is still returned.
- `ki_from_ic50` raises a `UserWarning` instead of a diagnostic.

### Statistics

The statistics behind the checks are returned, for applying your own multiple-comparison
correction across datasets. `GlobalFitResult.statistics_per["dataset name"]` gives them
per dataset.

```python
res = fit(conc, signal)
for s in res.statistics:
    print(s.name, s.statistic, s.p_value)
```

| Name | Test | Raises |
|---|---|---|
| `model_vs_constant` | F-test against a constant | `NO_FIT` at *P* ≥ 0.01 |
| `residual_runs` | One-sided runs test (Wald–Wolfowitz) | `RESIDUAL_STRUCTURE` at *P* < 0.025 |
| `residual_sign_test` | Exact two-sided sign test, replacing `residual_runs` when every residual shares a sign | `RESIDUAL_STRUCTURE` regardless of *P* |
| `residual_autocorrelation` | Lag-1 autocorrelation (`p_value` is `None`) | `RESIDUAL_STRUCTURE` above 0.3 |
| `heteroscedasticity` | One-sided Spearman test | `HETEROSCEDASTIC` at *P* < 0.01 |

## Plotting

| Method | Returns |
|---|---|
| `res.curve(conc_min, conc_max, n)` | The fitted curve `(x, y)`, log-spaced |
| `res.predict(conc)` | The fitted value at an arbitrary concentration |
| `res.residuals(conc, signal)` | Residuals (observed minus fitted) |

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(*res.curve(), "-")
ax.plot(conc, signal, "o")
ax.set_xscale("log")
```

`examples/plot_fit.py` draws the points, the fitted curve, and the residuals. It is not
part of the distribution; run it from a source checkout with the development
dependencies installed.

```bash
uv run python examples/plot_fit.py examples/titration_good.csv --unit nM --out examples/fit_good.png
```

![langmuir fit of examples/titration_good.csv](examples/fit_good.png)
