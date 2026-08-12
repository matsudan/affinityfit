# affinityfit

[![PyPI](https://img.shields.io/pypi/v/affinityfit)](https://pypi.org/project/affinityfit/)
[![Python versions](https://img.shields.io/pypi/pyversions/affinityfit)](https://pypi.org/project/affinityfit/)
[![ci](https://github.com/matsudan/affinityfit/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/matsudan/affinityfit/actions/workflows/ci.yml)

🚧 This library is currently under development and may change substantially.

A Python library that fits *K*<sub>d</sub> and related parameters from concentration and
signal data, and diagnoses whether the measurements constrain the resulting estimate.

Any observable that is linear in the fraction bound follows the same model (a saturation
curve), regardless of the measurement technique. Nuclear magnetic resonance (NMR) peak
intensity, surface plasmon resonance (SPR) steady-state response, and initial enzyme
velocity are all examples.

```
signal = baseline + Bmax * [L] / (Kd + [L])
```

Several datasets can be fitted simultaneously, sharing parameters across them or holding
some constant. A curve whose measured range does not bracket *K*<sub>d</sub> is
undetermined on its own, but sharing a parameter with a better-sampled dataset can
render the estimate identifiable.

## Usage

```bash
uv add affinityfit
```

```python
from affinityfit import DiagnosticCode, fit, load_csv

conc, signal = load_csv("titration.csv")
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

`res.diagnostics` is the canonical diagnostics API. Each immutable `Diagnostic`
has a stable `code` for program logic, a `severity` of `"warning"` or `"note"`, and a
concise English `message` for display. Do not branch on `message`: its prose may be
refined without changing the code. `res.warnings` and `res.notes` are severity-filtered
views containing the same `Diagnostic` objects.

`DiagnosticCode` is an enum listing every code this library can emit, so
`list(DiagnosticCode)` enumerates them without reading the source.
Each member is a plain string (`DiagnosticCode.NOT_SATURATED == "not_saturated"`), so
comparing `diagnostic.code` against either the enum member or the equivalent string
literal works the same way.

The input CSV has concentration in the first column and signal in the second. Header
rows, comment rows, and blank rows are skipped. Repeated rows at the same concentration
are kept as separate data points rather than averaged; passing them to the bootstrap as
replicates requires the separate `replicates` argument described below.

Missing values (blank cells), `nan`, and `inf` raise an error naming the line, as do a
negative concentration and a file with fewer than three numeric rows. Silently dropping
a blank row would change the number of points, and with it the degrees of freedom and
the diagnostics.

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

Output of the optional `report()` renderer:

```
model    : langmuir  (1:1 binding: signal = baseline + Bmax * L / (Kd + L))
interval : profile
Kd       = 10.1 +/- 0.8 nM
Bmax     = 0.995 +/- 0.017
baseline = 0.018 +/- 0.014
R^2      = 0.9998   (n = 8)
AICc     = -71.12   (AIC = -77.12)

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

`fit_diagnostics` contains findings about the complete fit, while
`diagnostics_per` keeps each dataset's findings scoped to its name. `result_for()`
combines those fit-wide and local records into its `FitResult`. `warnings` and
`notes` are severity-filtered views over both scopes at once; `warnings_per` and
`notes_per` cover the per-dataset findings only, so reading those alone misses
anything in `fit_diagnostics`.

## Weighting

By default every point counts equally, which assumes the measurement error is the same
size everywhere. That holds for a signal measured in resonance units, as in SPR, but not
for fluorescence, luminescence, or absorbance, where the error scales with the signal.
Since a saturation curve moves the signal from baseline to baseline + *B*<sub>max</sub>,
the absolute error near saturation can be tens of times larger than at low
concentration.

```python
res = fit(conc, signal, sigma=0.01 + 0.10 * signal)  # per-point standard deviation
```

`sigma` is treated as relative; the overall scale is still estimated from the
residuals, so the result is unchanged under a constant rescaling. In a global fit,
`Dataset(..., sigma=...)` can be given per dataset, which also sets the relative
weight between datasets (for example, one dataset ten times noisier than another).

`sigma` must be given for either every dataset or none of them. Giving it to only
some would leave the others with an implicit weight of 1, so the relative weight between
datasets would depend on the absolute scale of `sigma` (whether it is written as a
fraction, a percentage, or ppm). Mixing the two raises an error.

Fitting heteroscedastic data without weights costs more than precision: it also narrows
the confidence interval. Over 200 simulated 12-point titrations with 30% proportional
error and *K*<sub>d</sub> near the top of the measured range, 95% interval coverage was
79% unweighted against 94% with `sigma` supplied.

The diagnostics warn when the size of the residuals grows with the fitted value.

## Confidence intervals

`ci=` selects one of three methods.

| Method | Description |
|---|---|
| `profile` (default) | Pins *K*<sub>d</sub> and refits everything else, taking the boundary where the residual sum of squares rises significantly (an F-test). The interval can be asymmetric, and a side that cannot be determined is returned as `None` rather than as a number, and formatted as a one-sided limit |
| `asymptotic` | Reads the interval off the curvature of the covariance matrix. The fastest option, but reports a finite two-sided interval even for unidentifiable data |
| `bootstrap` | Resamples the data and refits to get the distribution of the estimate directly. Resamples replicates when given, residuals otherwise |

`replicates` has shape `(n_replicates, n_points)`.

```python
res = fit(conc, signal, ci="bootstrap", replicates=reps, n_boot=2000)
```

`Dataset` also accepts replicates on their own (`signal` is then their mean).

```python
Dataset("oxidized", conc, replicates=reps)
```

Under `ci="bootstrap"`, `n_boot` must be at or above the minimum needed for a percentile
interval (100), or it raises an error; below that, no interval can be formed.

The reported precision follows the uncertainty. A measurement with a 17% relative error
does not warrant three significant figures, so `Kd = 4.70e-8` is reported as
`(4.7 +/- 0.8)e-08` instead.

## Model

| Model | Parameters | Use |
|---|---|---|
| `langmuir` (default) | kd, bmax, baseline | 1:1 binding. `signal = baseline + Bmax·L/(Kd+L)` |
| `hill` | kd, bmax, baseline, n | Cooperativity, judged by whether the confidence interval of *n* contains 1. Read the caveat below before claiming *n* > 1 |
| `michaelis` | km, vmax, baseline | Enzyme kinetics. Same equation as langmuir, but *K*<sub>m</sub> is not an affinity |
| `ic50` | ic50, bmax, baseline, hillslope | Dose–response (four-parameter logistic, 4PL). A negative `bmax` inhibits, a positive one gives an EC<sub>50</sub> curve |
| `tight_binding` | kd, bmax, baseline, rt | 1:1 binding solved for ligand depletion, for a receptor not dilute against *K*<sub>d</sub> |

`bmax` in `langmuir` and `hill` is not restricted in sign. For an observable that
decreases on binding (fluorescence quenching, a drop in intensity), *K*<sub>d</sub>
retains its meaning and the decrease is expressed as a negative response coefficient in
the same equation. `vmax` in `michaelis` is a rate and stays non-negative, so use
`langmuir` or `hill` for decreasing data. A model of the wrong sign yields a
diagnostic rather than a plausible-looking number.

`Km = (k_off + k_cat)/k_on`, and only equals *K*<sub>d</sub> when `k_cat` is negligible.

*K*<sub>d</sub>, *K*<sub>m</sub>, and IC<sub>50</sub> are optimised on a logarithmic
scale. On a linear scale the fitted result depends on the unit of concentration: a
picomolar affinity (*K*<sub>d</sub> = 1 × 10<sup>−12</sup> M) expressed in molar units
comes out about 6700% too high on a 12-point titration. On the logarithmic scale
*K*<sub>d</sub> is recovered to within 10<sup>−6</sup> relative error across 18 orders
of magnitude, from fM to 100 M, and the same experiment written in M, nM, or pM gives
the same answer.

### Dose–response and *K*<sub>i</sub>

`ic50` is the four-parameter logistic. It is the same equation as `hill`, under the
names used to report a dose–response curve, and without the cooperativity checks, since
a slope near 1 is the ordinary case in a dose–response experiment rather than a finding.

```python
from affinityfit import ic50

res = fit(conc, response, model=ic50, unit="nM")
print(res.params["ic50"], res.params["hillslope"])
```

A displacement IC<sub>50</sub> is not a property of the competitor alone: raising the
concentration of the labelled ligand being displaced raises the IC<sub>50</sub> with it.
Dividing by `1 + [tracer]/Kd_tracer` removes that dependence and makes *K*<sub>i</sub>
comparable between assays run at different tracer concentrations.

```python
from affinityfit import Interval, ki_from_ic50

ki = ki_from_ic50(res, tracer_conc=5.0, tracer_kd=Interval(point=2.0, lower=1.6, upper=2.4))
print(ki.format("nM"))
```

Pass the whole result rather than one interval out of it, and give the tracer constant
as an `Interval` when its uncertainty is known.

**The standard form assumes a slope of 1.** It is derived for a single site under
competition, and a fitted slope whose interval excludes 1 indicates that the derivation
does not apply. The modified forms that cover a slope away from 1 raise the terms to
powers and [do not agree with one another](https://pubmed.ncbi.nlm.nih.gov/12481843/),
so the choice between them is left to the caller; the library reports the violation
rather than applying one silently. Only the `FitResult` carries the IC<sub>50</sub> and
the slope together, so only that overload can check it. Passing `res.intervals["ic50"]`
computes the same number without the check.

**The tracer constant's own error usually dominates.** With `r = [T]/Kd*`, the relative
error of `Kd*` propagates to *K*<sub>i</sub> damped by `r/(1+r)`, so at `[T] = 2.5·Kd*`
a 20% uncertainty on `Kd*` puts 14% onto *K*<sub>i</sub>, against about 5% from a
well-measured IC<sub>50</sub>. Treating `Kd*` as exact here reports ±5% where the
experiment supports ±15%. The two are independent, so they combine in quadrature, each
side of the interval separately; asymmetry is preserved, an undetermined limit stays
undetermined, and a lower limit driven past zero is reported as undetermined rather than
as a *K*<sub>i</sub> of zero. `tracer_conc` is taken as exact.

For competitive enzyme inhibition the same expression applies with `[S]` and `Km`.

The relation also assumes the competitor and the tracer exclude each other from one
site, and that the free tracer concentration is close to the total. A fit cannot check
either assumption, because the curve has the same shape in both cases.

### Ligand depletion

Every model in the table except `tight_binding` assumes the free ligand concentration
equals the total one. This assumption fails once the receptor is not much more dilute
than *K*<sub>d</sub>, because each molecule bound is one fewer left in solution. The
curve still looks like a saturation curve, and the fit statistics do not reveal the
problem. On noiseless 1:1 data with the receptor at five times *K*<sub>d</sub> (15
concentrations spanning 10<sup>−2</sup> to 10<sup>2</sup> times *K*<sub>d</sub> plus a
blank, baseline held at 0), `langmuir` reports *K*<sub>d</sub> = 3.99 against a true
1.0, at *R*<sup>2</sup> = 0.9919.

`tight_binding` solves the 1:1 equilibrium without that assumption. Pass the total
receptor concentration as `rt`, normally as a constant.

```python
from affinityfit import tight_binding

res = fit(conc, signal, model=tight_binding, fixed={"rt": 5.0}, unit="uM")
```

When `rt` is left free instead, it is estimated from the data and measures the active
concentration: the fraction of the immobilised or pipetted receptor that is actually
binding. Depletion changes the shape of the curve and not only its midpoint, which
prevents `rt` and *K*<sub>d</sub> from trading off against each other.

```python
res = fit(conc, signal, model=tight_binding, unit="uM")
print(res.intervals["rt"].format("uM"))
```

The diagnostic that recommends this model (see below) is suppressed once the model is in
use. Two limitations apply:

- `fixed=` applies one value to every dataset, so in a global fit spanning several
  receptor concentrations `rt` cannot be fixed per dataset. Fit those separately.
- The checks on where the measurements sit compare the range against *K*<sub>d</sub>.
  Depletion reaches the plateau at a lower multiple of *K*<sub>d</sub> than a hyperbola
  does, so those thresholds are conservative here and may still call for higher
  concentrations than this model requires.

### Cooperativity under depletion

Depletion does not only shift the curve, it steepens it, and `hill` reads that steepness
as an exponent. The same data as above, with no cooperativity anywhere in the system,
fits `hill` at *n* = 1.43 [1.32, 1.54], *R*<sup>2</sup> = 0.9992. The interval excludes
1, so the check described above (`intervals["n"].contains(1.0)`) reports cooperativity
that does not exist.

No model here solves depletion and cooperativity at once. Adding one would impose an
exact conservation law on the Hill equation, which is phenomenological to begin with.
Cooperativity must therefore be measured where depletion is absent: keep the receptor at
or below *K*<sub>d</sub>/10 for that question.

An *n* significantly above 1 is always reported together with the alternatives that
produce the same shape (depletion, self-association, or a reading taken before
equilibrium). Passing `receptor_conc=` allows the first of them to be checked rather
than only listed; without it, the warning states that the check could not be performed.

## Model selection

Use `aicc` (the corrected Akaike information criterion) to compare models or
parameter-sharing schemes. `aic` is kept for reference, but the uncorrected AIC is only
asymptotically valid and favours the model with more parameters at the scale of a
typical titration (6–15 points, 3–6 parameters, *n*/*k* of 2–5).

```python
shared.aicc < free.aicc
fit_global(ds, model=hill).aicc < fit_global(ds, model=langmuir).aicc
```

`report()` shows AICc first, with AIC in parentheses for reference. When the sample
is too small for the correction to be defined (*n* − *k* − 1 ≤ 0), `aicc` is infinite,
and `FitResult.report()` drops the line instead of printing an infinity.

## Diagnostics

*K*<sub>d</sub> is frequently undetermined even when *R*<sup>2</sup> exceeds 0.99. The
conditions below are detected automatically. A quantity whose name changes by model
(*K*<sub>d</sub>, *K*<sub>m</sub>) is shown under that model's own label.

The diagnostics fall into two groups: whether the fit itself is sound, and whether the
measurements are placed well.

`Diagnostic.code` (a `DiagnosticCode` member) is the value to branch on; see
[`DiagnosticCode`](#usage) for how to list every code from Python.

| Condition | Meaning | Code |
|---|---|---|
| Amplitude within 1% of the signal range | The fit is effectively a flat line; the model cannot express the shape of the data | `AMPLITUDE_COLLAPSED` |
| *R*<sup>2</sup> < 0.5 | The model does not capture the trend in the data; the value of *K*<sub>d</sub> is meaningless | `NO_FIT` |
| *R*<sup>2</sup> < 0.9 | Low for a saturation curve; either noise or the wrong model | `POOR_FIT` |
| Systematic sign in the residuals | The shape of the model does not match the mechanism, even with a high coefficient of determination. Needs at least eight points | `RESIDUAL_STRUCTURE` |
| A parameter stuck at a bound | That value is an artefact of the constraint, not an estimate, and cannot be reported | `PARAM_AT_BOUND` |
| Highest concentration < 3 × *K*<sub>d</sub> | Saturation not reached; *K*<sub>d</sub> and *B*<sub>max</sub> cannot be mathematically separated, so the conclusion is limited to "*K*<sub>d</sub> > highest concentration" | `NOT_SATURATED` |
| Highest concentration < 10 × *K*<sub>d</sub> | The estimate of *B*<sub>max</sub> is unstable, and the confidence interval on *K*<sub>d</sub> widens as well | `WEAKLY_SATURATED` |
| Data points < 2 × estimated parameters | Not enough information; confidence intervals are indicative only (fixed parameters are not counted) | `FEW_POINTS` |
| No points near *K*<sub>d</sub> (*K*<sub>d</sub>/3 to 3 × *K*<sub>d</sub>) | The inflection point of the curve is underdetermined; adding points here improves the estimate the most | `NO_POINTS_NEAR_KD` |
| One point near *K*<sub>d</sub> (*K*<sub>d</sub>/3 to 3 × *K*<sub>d</sub>) | The inflection point of the curve rests on a single point | `ONE_POINT_NEAR_KD` |
| Lowest concentration > *K*<sub>d</sub> | Every point sits on the saturated side; *K*<sub>d</sub> is set by extrapolation and should not be reported to extra significant figures | `KD_EXTRAPOLATED` |
| No points at or below *K*<sub>d</sub>/10 | baseline is estimated together with the curve, which can shift *B*<sub>max</sub> | `NO_LOW_CONC` |
| Receptor concentration > *K*<sub>d</sub>/10 | Ligand depletion causes *K*<sub>d</sub> to be overestimated; switch to `tight_binding`. Not reported when that model is already in use. With `hill`, also states that *n* is inflated and cooperativity cannot be judged at all under these conditions | `LIGAND_DEPLETION` |
| The confidence interval of the Hill coefficient *n* is undetermined | The residuals leave no scatter from which to determine a direction, or a side is undetermined; cooperativity cannot be judged | `HILL_N_UNDETERMINED` |
| The confidence interval of the Hill coefficient *n* contains 1 | Cooperativity cannot be claimed | `HILL_N_INCLUDES_ONE` |
| Hill coefficient *n* significantly > 1 | Positive cooperativity is one reading, but depletion, self-association, and a pre-equilibrium reading give the same shape. States when `receptor_conc` was not supplied to check the first of them | `HILL_N_ABOVE_ONE` |
| Hill coefficient *n* significantly < 1 | Negative cooperativity, heterogeneous sites, or a heterogeneous sample | `HILL_N_BELOW_ONE` |
| Residual size proportional to the fitted value | Heteroscedastic error; omitting `sigma` narrows the interval. Needs at least eight points, and is not checked once `sigma` is supplied | `HETEROSCEDASTIC` |
| Zero degrees of freedom (points ≤ parameters) | The curve passes through every point by construction; no confidence interval can be computed | `NO_DEGREES_OF_FREEDOM` |
| Rank-deficient Jacobian | Parameters cannot be distinguished, so the values are not uniquely determined; more concentrations are needed | `RANK_DEFICIENT_JACOBIAN` |
| One side of a confidence interval is undetermined | The point estimate should not be reported; sharing or a wider measured range is needed. The Hill coefficient is excluded, the `HILL_N_*` codes covering it instead | `LIMIT_UNDETERMINED` |
| Too many bootstrap resamples failed to converge | Below the minimum for a percentile interval; the interval is reported as undetermined | `BOOTSTRAP_INSUFFICIENT_SAMPLES` |
| Some bootstrap resamples failed to converge | The interval may be narrower than it should be, since the resamples that converge are the easier ones to fit | `BOOTSTRAP_FAILURES` |
| Amplitude shared in a global fit rescues an unsaturated dataset | Sharing rendered an otherwise unidentifiable estimate identifiable; not a problem | `SHARED_AMPLITUDE_IDENTIFIES_LOCATION` |
| Amplitude free in a global fit with an unsaturated dataset | Consider sharing the amplitude if the maximum signal is common across datasets | `UNSHARED_AMPLITUDE` |

`ki_from_ic50` raises a `UserWarning` rather than a diagnostic, since it is called
after the fit and has no `FitResult` of its own to attach one to.

In a global fit, findings whose premise is removed by sharing or fixing are suppressed.
If `bmax` is shared, for example, `NOT_SATURATED` no longer applies and
`SHARED_AMPLITUDE_IDENTIFIES_LOCATION` is emitted in its place. That note is
withheld when the fit itself is broken (*R*<sup>2</sup> < 0.5, or a collapsed
amplitude), since it would otherwise contradict the other findings.

In `examples/titration_unsaturated.csv`, *R*<sup>2</sup> = 0.9936 suggests a good fit,
but the highest concentration is only a third of *K*<sub>d</sub>, so the upper limit is
undetermined and the result can only be reported as `Kd > 76 nM`.

When a confidence interval cannot be computed (zero degrees of freedom, a rank-deficient
Jacobian, an inner refit that fails to converge, or too few bootstrap resamples
succeeding), no value is substituted; the interval is reported as undetermined, with the
reason given. The point estimate itself is still returned.

Residuals are tested with both a runs test (Wald–Wolfowitz) and lag-1 autocorrelation,
and a warning is emitted when either crosses its threshold. Over 2000 simulated 15-point
titrations fitted with the correct model, the false-positive rate was 1.1%.

### Statistics

The warnings are judged at a fixed significance level. Stacking tests across several
datasets raises the false-positive rate accordingly, so the underlying statistic and *P*
value are returned as-is.

```python
res = fit(conc, signal)
for s in res.statistics:
    print(s.name, s.statistic, s.p_value)
```

For several datasets, these are available per dataset through
`GlobalFitResult.statistics_per["dataset name"]`. The names are `residual_runs`,
`residual_autocorrelation`, and `heteroscedasticity`, plus `residual_sign_test`,
which stands in for `residual_runs` when every residual shares a sign and the runs count
is degenerate. `residual_autocorrelation` is judged against a fixed threshold rather
than a *P* value, so its `p_value` is `None`.

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

Keep the concentration axis logarithmic. On a linear axis the low-concentration region
collapses, hiding both the curvature near *K*<sub>d</sub> and design issues such as the
measurements being skewed toward saturation.

`examples/plot_fit.py` is a sample that draws the measured points, the fitted
curve, and a residual panel in a single figure. It is not part of the distribution.

```bash
uv run python examples/plot_fit.py examples/titration_good.csv --unit nM
```

Running this sample on `examples/titration_good.csv`:

![langmuir fit of examples/titration_good.csv](examples/fit_good.png)
