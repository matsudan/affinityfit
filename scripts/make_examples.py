"""Regenerate the example datasets and the figure in the README.

The example CSVs used to be typed by hand, which made the "good" one inconsistent with
the model it was supposed to illustrate: a Hill fit put the coefficient at 1.076 with an
interval excluding 1, so the file that demonstrates a clean 1:1 titration was in fact
evidence of cooperativity. Generating them from the model with a fixed seed keeps the
examples honest and reproducible.

    uv run python scripts/make_examples.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from bindfit import langmuir

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# よく条件づけられた滴定: Kd を挟んで 1/10 から 100 倍まで測る。
GOOD = {
    "path": EXAMPLES / "titration_good.csv",
    "conc": np.array([0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]),
    "kd": 10.0,
    "bmax": 1.0,
    "baseline": 0.02,
    "noise": 0.01,
    "seed": 20260810,
}

# 飽和に達していない滴定: 最高濃度が Kd の 1/3 しかない。決定係数は高いのに Kd の
# 信頼区間は広く、片側が決まらないこともある、という例。
UNSATURATED = {
    "path": EXAMPLES / "titration_unsaturated.csv",
    "conc": np.array([0.0, 1.0, 3.0, 10.0, 30.0, 100.0]),
    "kd": 300.0,
    "bmax": 1.0,
    "baseline": 0.02,
    "noise": 0.01,
    "seed": 20260811,
}


def write(spec: dict) -> Path:
    conc = spec["conc"]
    signal = langmuir(conc, spec["kd"], spec["bmax"], spec["baseline"])
    signal = signal + np.random.default_rng(spec["seed"]).normal(0.0, spec["noise"], conc.size)

    path: Path = spec["path"]
    lines = [
        "# Synthetic data, not from any publication. Regenerate with scripts/make_examples.py",
        f"# langmuir: Kd={spec['kd']:g} nM, Bmax={spec['bmax']:g}, baseline={spec['baseline']:g}, "
        f"gaussian noise sd={spec['noise']:g}, seed={spec['seed']}",
        "concentration_nM,signal",
    ]
    lines += [f"{c:g},{s:.4f}" for c, s in zip(conc, signal, strict=True)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    for spec in (GOOD, UNSATURATED):
        print(f"wrote {write(spec)}")

    figure = EXAMPLES / "fit_good.png"
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(EXAMPLES / "plot_fit.py"),
            str(GOOD["path"]),
            "--unit",
            "nM",
            "--out",
            str(figure),
        ],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
