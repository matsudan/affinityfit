"""Smoke check for an installation with runtime dependencies only.

Run by the `runtime-deps-only` job in CI. It is a file rather than a here-document
inside the workflow so that ruff and ty check it; an inline script in YAML rots
silently when the API changes.

Two things are verified.

1. matplotlib is genuinely absent, and the library works anyway. Plotting lives in
   `examples/plot_fit.py` and matplotlib is a development dependency, so a consumer
   gets numpy and scipy and nothing else. The unit test for this stubs `sys.modules`,
   which is not the same as an environment where the package was never installed.
2. The PEP 561 marker ships inside the installed package. Without it mypy reads no
   annotations at all, so a consumer's type checking silently turns into a no-op.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

import affinityfit
from affinityfit import Dataset, fit, fit_global, hill, ic50, ki_from_ic50, langmuir, tight_binding


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}", file=sys.stderr)
        raise SystemExit(1)


def main() -> int:
    require(
        importlib.util.find_spec("matplotlib") is None,
        "matplotlib が入っています。ライブラリ本体が描画に依存していないか確認してください。",
    )

    conc = np.array([0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])

    single = fit(conc, langmuir(conc, 10.0, 1.0, 0.0))
    require(abs(single.params["kd"] - 10.0) < 1e-6, f"kd を回収できません: {single.params}")
    require(len(single.curve()[0]) == 300, "curve() が 300 点を返しません")

    shared = fit_global(
        [
            Dataset("a", conc, langmuir(conc, 10.0, 1.0, 0.0)),
            Dataset("b", conc, langmuir(conc, 40.0, 1.0, 0.0)),
        ],
        shared=["bmax"],
        fixed={"baseline": 0.0},
    )
    require(
        abs(shared.params["b"]["kd"] - 40.0) < 1e-3,
        f"共有パラメータつきの推定が合いません: {shared.params}",
    )

    cooperative = fit(conc, hill(conc, 10.0, 1.0, 0.0, 2.0), model=hill)
    require(abs(cooperative.params["n"] - 2.0) < 1e-3, f"Hill 係数が合いません: {cooperative.params}")

    dose = fit(conc[1:], ic50(conc[1:], 10.0, -100.0, 100.0, 1.0), model=ic50)
    require(abs(dose.params["ic50"] - 10.0) < 1e-3, f"IC50 が合いません: {dose.params}")

    # The quadratic form leans on numpy alone, and `conversions` is imported for the first time here,
    # so both are exercised in an environment where nothing beyond numpy and scipy is installed.
    depleted = fit(conc, tight_binding(conc, 10.0, 1.0, 0.0, 50.0), model=tight_binding, fixed={"rt": 50.0})
    require(abs(depleted.params["kd"] - 10.0) < 1e-3, f"tight_binding の Kd が合いません: {depleted.params}")
    require(abs(ki_from_ic50(10.0, tracer_conc=2.0, tracer_kd=2.0) - 5.0) < 1e-12, "Cheng-Prusoff 補正が合いません")

    marker = pathlib.Path(affinityfit.__file__).with_name("py.typed")
    require(marker.is_file(), f"py.typed が配布物にありません: {marker}")

    print("ok: fit / fit_global / hill / ic50 / tight_binding が動作し、py.typed も同梱されています")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
