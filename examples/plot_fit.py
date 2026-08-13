"""Sample: read a titration from CSV, fit it, and draw the figure yourself.

This is a sample, not part of the package. affinityfit deliberately returns numbers
rather than figures, because a publication figure always needs its own fonts,
colours and panel layout. What is worth copying from here is the plotting geometry:

- a logarithmic concentration axis, since linear spacing collapses the low end and
  hides the curvature around the half-saturation point
- a point at zero concentration drawn separately, as it cannot sit on a log axis
- a residual panel underneath, which is where a shape mismatch becomes visible

Run it with matplotlib available (it is a development dependency, not a dependency
of the library):

    uv run python examples/plot_fit.py examples/titration_good.csv --unit nM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from affinityfit import MODELS, FitResult, fit


def read_csv(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read `(concentration, signal)` from a two-column CSV file.

    Comment lines starting with `#` and blank lines are skipped. A header row is
    detected because it parses to `NaN` and is dropped only when it is the first row;
    a missing or non-numeric cell elsewhere becomes `NaN`, which `fit()` rejects with
    the offending row index rather than silently dropping it.
    """
    data = np.genfromtxt(path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if np.isnan(data[0]).all():
        data = data[1:]
    return data[:, 0], data[:, 1]


def plot_fit(
    conc: NDArray[np.float64],
    signal: NDArray[np.float64],
    result: FitResult,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Draw the measurements, the fitted curve and the residuals into one PNG.

    Args:
        conc: Ligand concentration.
        signal: Observed signal.
        result: The fit to draw.
        out_path: Path of the PNG file to write.
        title: Title of the upper panel.

    Returns:
        The path that was written.

    Raises:
        ValueError: If no positive concentration is present.
    """
    conc = np.asarray(conc, dtype=float)
    signal = np.asarray(signal, dtype=float)

    positive = conc[conc > 0]
    if positive.size == 0:
        raise ValueError("No positive concentration values found.")
    zero_position = min(positive.min() / 3.0, result.location / 1000.0)
    x_plot = np.where(conc > 0, conc, zero_position)

    curve_x, curve_y = result.curve(zero_position, positive.max() * 3.0)

    fig, (ax, ax_residual) = plt.subplots(
        2,
        1,
        figsize=(6.4, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    model = result.model
    ax.plot(curve_x, curve_y, "-", color="tab:blue", lw=1.8, label=f"{model.name} fit")
    at_zero = conc == 0
    ax.plot(x_plot[~at_zero], signal[~at_zero], "o", color="k", ms=5, label="measured")
    if at_zero.any():
        ax.plot(x_plot[at_zero], signal[at_zero], "s", mfc="none", color="k", ms=6, label="conc = 0")

    unit = f" [{result.unit}]" if result.unit else ""
    location = result.location
    baseline = result.params[model.baseline] if model.baseline else 0.0
    amplitude = result.params[model.amplitude]
    ax.axvline(location, color="tab:red", ls="--", lw=1.0)
    ax.annotate(
        f"{model.label(model.location)} = {result.intervals[model.location].format(result.unit)}",
        xy=(location, baseline + amplitude / 2),
        xytext=(6, -4),
        textcoords="offset points",
        color="tab:red",
        fontsize=9,
        va="top",
    )
    ax.axhline(baseline + amplitude, color="gray", ls=":", lw=1.0)
    ax.set_ylabel("signal")
    ax.set_title(title or f"{model.name} fit  (R^2 = {result.r_squared:.4f})")
    ax.legend(frameon=False, fontsize=9, loc="best")

    ax_residual.axhline(0, color="gray", lw=1.0)
    ax_residual.plot(x_plot, result.residuals(conc, signal), "o", color="k", ms=4)
    ax_residual.set_xscale("log")
    ax_residual.set_xlabel(f"ligand concentration{unit} (log scale)")
    ax_residual.set_ylabel("residual")

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path, help="1 列目=濃度, 2 列目=シグナル の CSV")
    parser.add_argument("--model", choices=sorted(MODELS), default="langmuir")
    parser.add_argument("--unit", default="", help="濃度の単位名（例: nM）")
    parser.add_argument("--out", type=Path, default=None, help="出力 PNG。既定は入力と同じ名前")
    args = parser.parse_args(argv)

    # Input errors (missing values, non-numeric cells, a negative concentration) are reported by fit() as a
    # ValueError naming the offending array index. There is no need to show a traceback, so only the message
    # is printed before exiting.
    try:
        conc, signal = read_csv(args.csv)
        result = fit(conc, signal, model=MODELS[args.model], unit=args.unit)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(result.report())

    out = args.out or args.csv.with_suffix(".png")
    print(f"\nsaved: {plot_fit(conc, signal, result, out, title=args.csv.name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
