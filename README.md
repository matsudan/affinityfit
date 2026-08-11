# affinityfit

[![PyPI](https://img.shields.io/pypi/v/affinityfit)](https://pypi.org/project/affinityfit/)
[![Python versions](https://img.shields.io/pypi/pyversions/affinityfit)](https://pypi.org/project/affinityfit/)
[![ci](https://github.com/matsudan/affinityfit/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/matsudan/affinityfit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/pypi/l/affinityfit)](https://github.com/matsudan/affinityfit/blob/main/LICENSE)

A Python library that fits **Kd** and related parameters from concentration and
signal data, and diagnoses whether the resulting estimate can be trusted.

Any observable that is linear in the fraction bound follows the same model (a
saturation curve), regardless of the measurement technique. NMR peak intensity,
SPR steady-state response, and initial enzyme velocity are all examples.

```
signal = baseline + Bmax * [L] / (Kd + [L])
```

Several datasets can be fitted simultaneously, sharing parameters across them or
holding some constant. A curve whose measured range does not bracket Kd is
otherwise undetermined on its own; sharing a parameter with a better-sampled
dataset can make the estimate identifiable.

## Usage

```bash
uv add affinityfit
```

```python
from affinityfit import fit, load_csv

conc, signal = load_csv("titration.csv")
res = fit(conc, signal, unit="nM")

print(res.report())
print(res.params["kd"], res.intervals["kd"].format("nM"))
for warning in res.warnings:
    print(warning)
```

The input CSV has concentration in the first column and signal in the second.
Header rows, comment rows, and blank rows are skipped. Repeated rows at the same
concentration are used as replicates as-is.

Missing values (blank cells), `nan`, and `inf` raise an error with the row number.
Silently dropping a blank row would change the number of points, and with it the
degrees of freedom and the diagnostics.

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
R^2      = 0.9998   (n = 8)
AICc     = -71.12   (AIC = -77.12)

診断チェック: 問題は検出されませんでした。
```

If the concentration of the fixed partner (a receptor, a lectin, and so on) is
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
print(res.report())

# Pull out one dataset. Its warnings and notes come along with it.
sub = res.result_for("oxidized")
print(sub.report())
x, y = sub.curve()
```

`warnings` are problems, `notes` are remarks such as "sharing is what made this
estimate possible". `GlobalFitResult.warnings` / `.notes` return everything with a
`[dataset name]` prefix, while `result_for()` returns the ones for that dataset
(plus any that concern the fit as a whole) without the prefix.

## Weighting

By default every point counts equally, which assumes the measurement error is the
same size everywhere. That holds for a system like SPR response units, but not for
fluorescence, luminescence, or absorbance, where the error scales with the signal.
Since a saturation curve moves the signal from baseline to baseline+Bmax, the
absolute error near saturation can be tens of times larger than at low
concentration.

```python
res = fit(conc, signal, sigma=0.01 + 0.10 * signal)  # per-point standard deviation
```

`sigma` is treated as relative; the overall scale is still estimated from the
residuals, so the result is unchanged under a constant rescaling. In a global fit,
`Dataset(..., sigma=...)` can be given per dataset, which also carries over the
relative weight between datasets (one being ten times noisier than another, say).

sigma must be given for **either every dataset or none of them**. Giving it to only
some would leave the others with an implicit weight of 1, so the relative weight
between datasets would be decided by the absolute scale of sigma (whether it is
written as a fraction, a percentage, or ppm). Mixing the two raises an error.

Fitting heteroscedastic data without weights costs more than precision: **it also
narrows the confidence interval** (measured for Kd=10 with 30% proportional error,
95% CI coverage was 79% unweighted versus 94% with sigma supplied).

The diagnostics warn when the size of the residuals grows with the fitted value.

## Confidence Intervals

`ci=` selects one of three methods.

| Method | Description |
|---|---|
| `profile` (default) | Pins Kd and refits everything else, taking the boundary where the residual sum of squares rises significantly (an F-test). The interval can be asymmetric; when one side cannot be pinned down it comes back as, for example, `Kd > 1.6` |
| `asymptotic` | Reads the interval off the curvature of the covariance matrix. The fastest option, but can claim a finite two-sided interval even for unidentifiable data |
| `bootstrap` | Resamples the data and refits to get the distribution of the estimate directly. Resamples replicates when given, residuals otherwise |

Reporting error from repeated experiments is standard practice in this field.
Passing replicate measurements follows that convention.

```python
res = fit(conc, signal, ci="bootstrap", replicates=reps, n_boot=2000)
```

`Dataset` also accepts replicates on their own (`signal` is then their mean).

```python
Dataset("oxidized", conc, replicates=reps)
```

`n_boot` must be at or above the minimum needed for a percentile interval (100),
or it raises an error; below that, no interval can be formed.

The reported precision follows the uncertainty. A measurement with a 17% relative
error does not warrant three significant figures, so `Kd = 4.70e-8` is reported as
`(4.7 +/- 0.8)e-08` instead.

## Model

| Model | Parameters | Use |
|---|---|---|
| `langmuir` (default) | kd, bmax, baseline | 1:1 binding. `signal = baseline + Bmax·L/(Kd+L)` |
| `hill` | kd, bmax, baseline, n | Cooperativity, judged by whether the confidence interval of n contains 1 |
| `michaelis` | km, vmax, baseline | Enzyme kinetics. Same equation as langmuir, but Km is not an affinity |

`bmax` in `langmuir` and `hill` is not restricted in sign. An observable that
**decreases** on binding (fluorescence quenching, a drop in intensity) still has
the same meaning for Kd, expressed as a negative response coefficient in the same
equation. `vmax` in `michaelis` is a rate and stays non-negative, so use `langmuir`
or `hill` for decreasing data. Fitting the wrong kind of model to the data warns
rather than returning a plausible-looking number.

`Km = (k_off + k_cat)/k_on`, and only equals Kd when `k_cat` is negligible.

Kd and Km are optimised on a logarithmic scale. Handled linearly, the fitted
result would depend on the unit of concentration, reaching a relative error of
6800% for a picomolar affinity (Kd = 1e-12 M) expressed in molar units. As it
stands, Kd is recovered to within 1e-6 relative error across 18 orders of
magnitude, from fM to 100 M, giving the same answer whether the same experiment is
written in M, nM, or pM.

## Model Selection

Use `aicc` (the corrected Akaike information criterion) to compare models or
parameter-sharing schemes. `aic` is kept for reference, but the uncorrected AIC is
only asymptotically valid and favours the model with more parameters at the scale
of a typical titration (6-15 points, 3-6 parameters, n/k of 2-5).

This has been confirmed empirically as well: with fewer points, the uncorrected
AIC tends to favour the model with more parameters.

```python
shared.aicc < free.aicc
fit_global(ds, model=hill).aicc < fit_global(ds, model=langmuir).aicc
```

When the sample is too small for the correction to be defined (n − k − 1 ≤ 0),
`aicc` is infinite. `report()` shows AICc first, with AIC in parentheses for
reference.

## Diagnostics

It is not unusual for Kd to be undetermined even when R² is above 0.99. The
following are warned about automatically. A quantity whose name changes by model
(Kd, Km) is shown under that model's own label.

There are two kinds of diagnostics: whether **the fit itself is sound**, and
whether **the measurements are placed well**.

| Condition | Meaning |
|---|---|
| Amplitude collapsed to near 0 | The fit is effectively a flat line; the model cannot express the shape of the data |
| R² < 0.5 | The model does not capture the trend in the data; the value of Kd is meaningless |
| R² < 0.9 | Low for a saturation curve; either noise or the wrong model |
| Systematic sign in the residuals | The shape of the model does not match the mechanism, even with a high coefficient of determination |
| A parameter stuck at a bound | That value is an artefact of the constraint, not an estimate, and cannot be reported |
| Highest concentration < 3 × Kd | Saturation not reached; Kd and Bmax cannot be mathematically separated, so the conclusion is limited to "Kd > highest concentration" |
| Highest concentration < 10 × Kd | The estimate of Bmax is unstable, and the confidence interval on Kd widens as well |
| Data points < 2 × estimated parameters | Not enough information; confidence intervals are indicative only (fixed parameters are not counted) |
| 0-1 points near Kd (Kd/3 to 3Kd) | The inflection point of the curve is underdetermined; adding points here helps the most |
| Lowest concentration > Kd | Every point sits on the saturated side; Kd is set by extrapolation and should not be reported to extra significant figures |
| No points at or below Kd/10 | baseline is estimated together with the curve, which can shift Bmax |
| Receptor concentration > Kd/10 | Ligand depletion causes Kd to be overestimated; a tight-binding (quadratic) model is needed |
| Hill coefficient n's CI contains 1 | Cooperativity cannot be claimed |
| Residual size proportional to the fitted value | Heteroscedastic error; omitting `sigma` narrows the interval |
| Zero degrees of freedom (points ≤ parameters) | The curve passes through every point by construction; no confidence interval can be computed |
| Rank-deficient Jacobian | Parameters cannot be told apart, so the values are not uniquely determined; more concentrations are needed |
| One side of a confidence interval is undetermined | The point estimate should not be reported; sharing or a wider measured range is needed |

In a global fit, remarks whose premise is removed by sharing or fixing are
suppressed. For example, if `bmax` is shared, "Kd and Bmax cannot be separated"
no longer applies, and a note saying "sharing is what made this estimate
possible" appears instead. That note is withheld, though, when the fit itself is
broken (R² < 0.5, or a collapsed amplitude), since it would otherwise contradict
the other advice.

`examples/titration_unsaturated.csv` is an example where R² = 0.9936 looks like a
good fit, but the highest concentration is only a third of Kd, so the upper limit
is undetermined and the result can only be reported as `Kd > 76 nM`.

When a confidence interval cannot be computed (zero degrees of freedom, a
rank-deficient Jacobian, an inner refit that fails to converge, or too few
bootstrap resamples succeeding), no number is invented; the interval is reported
as undetermined, with the reason given. The point estimate itself is still
returned.

Residuals are tested with both a runs test (Wald-Wolfowitz) and lag-1
autocorrelation, and a warning fires if either is significant. The false-positive
rate is kept under 2%.

### Statistics

The warnings are judged at a fixed significance level. Stacking tests across
several datasets raises the false-positive rate accordingly, so the underlying
statistic and p-value are returned as-is.

```python
res = fit(conc, signal)
for s in res.statistics:
    print(s.name, s.statistic, s.p_value)
```

For several datasets, these are available per dataset through
`GlobalFitResult.statistics_per["dataset name"]`. `residual_autocorrelation` is
judged against a fixed threshold rather than a p-value, so its `p_value` is
`None`.

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

Keep the concentration axis logarithmic. On a linear axis the low-concentration
region collapses, hiding both the curvature near Kd and design issues such as the
measurements being skewed toward saturation.

`examples/plot_fit.py` is a sample that draws the measured points, the fitted
curve, and a residual panel in a single figure. It is not part of the
distribution.

```bash
uv run python examples/plot_fit.py examples/titration_good.csv --unit nM
```

Running this sample on `examples/titration_good.csv`:

![langmuir fit of examples/titration_good.csv](examples/fit_good.png)
