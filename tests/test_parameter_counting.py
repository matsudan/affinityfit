"""How parameters are counted, and the equality and hashing of `Dataset`.

The count behind `few_points` agrees with the degrees of freedom `fit_global` uses (n - n_slots).
A `Dataset` holds arrays, so equality comparison and hashing must not raise.
"""

from __future__ import annotations

import numpy as np
import pytest

from affinityfit import Dataset, fit, fit_global, hill, langmuir


def titration(n_points, kd=10.0):
    conc = np.logspace(-1, 2, n_points)
    return conc, langmuir(conc, kd, 1.0, 0.0)


def few_points_diagnostics(result):
    return [diagnostic for diagnostic in result.warnings if diagnostic.code == "few_points"]


# ------------------------------ few_points counts only the estimated parameters


@pytest.mark.parametrize(
    ("n_points", "fixed", "should_fire"),
    [
        (3, {"baseline": 0.0}, True),  # 2 estimated, 3 < 4
        (4, {"baseline": 0.0}, False),  # 2 estimated, 4 >= 4
        (5, {"baseline": 0.0}, False),
        (5, {}, True),  # 3 estimated, 5 < 6
        (6, {}, False),
    ],
)
def test_few_points_counts_only_estimated_parameters(n_points, fixed, should_fire):
    conc, signal = titration(n_points)
    result = fit(conc, signal, fixed=fixed)
    assert bool(few_points_diagnostics(result)) is should_fire


def test_few_points_diagnostics_have_warning_severity():
    conc, signal = titration(3)
    result = fit(conc, signal, fixed={"baseline": 0.0})
    assert few_points_diagnostics(result)[0].severity == "warning"


def test_two_fixed_parameters_are_both_excluded():
    conc = np.logspace(-1, 2, 5)
    signal = hill(conc, 10.0, 1.0, 0.0, 2.0)
    result = fit(conc, signal, model=hill, fixed={"baseline": 0.0, "n": 2.0})
    assert not few_points_diagnostics(result)  # 2 estimated, 5 >= 4


def test_the_count_matches_the_degrees_of_freedom_used_for_the_fit():
    """The count used by the diagnostics and the degrees of freedom used by `fit_global` are the same number."""
    conc = np.logspace(-1, 2, 5)
    signal = langmuir(conc, 10.0, 1.0, 0.0)
    result = fit_global([Dataset("d", conc, signal)], fixed={"baseline": 0.0}, ci="asymptotic")
    assert result.n_free_params == 2
    message = few_points_diagnostics(result.result_for("d"))
    assert not message  # 5 >= 2 * 2


def test_hill_without_fixed_parameters_still_fires_at_the_right_size():
    conc = np.logspace(-1, 2, 7)
    signal = hill(conc, 10.0, 1.0, 0.0, 2.0)
    result = fit(conc, signal, model=hill)
    assert few_points_diagnostics(result)  # 4 estimated, 7 < 8


# ---------------------------------- Equality and hashing of Dataset


def test_dataset_equality_does_not_raise():
    conc, signal = titration(6)
    first = Dataset("a", conc, signal)
    second = Dataset("a", conc, signal)
    assert first == first
    assert first != second  # compared by identity


def test_dataset_is_hashable():
    conc, signal = titration(6)
    dataset = Dataset("a", conc, signal)
    assert isinstance(hash(dataset), int)
    assert hash(dataset) == hash(dataset)


def test_dataset_works_in_a_set_and_as_a_dict_key():
    conc, signal = titration(6)
    first = Dataset("a", conc, signal)
    second = Dataset("b", conc, signal)
    assert len({first, second, first}) == 2
    mapping = {first: "oxidized", second: "reduced"}
    assert mapping[first] == "oxidized"


def test_dataset_with_replicates_is_also_hashable():
    conc, signal = titration(6)
    replicates = np.vstack([signal, signal + 0.01])
    dataset = Dataset("a", conc, signal, replicates=replicates)
    assert isinstance(hash(dataset), int)


def test_dataset_still_validates_its_input():
    """Dropping the equality comparison leaves the input validation working."""
    with pytest.raises(ValueError, match="different lengths"):
        Dataset("a", np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="NaN or infinity"):
        Dataset("a", np.array([1.0, 2.0, 3.0]), np.array([1.0, np.nan, 3.0]))


def test_dataset_is_still_frozen():
    conc, signal = titration(6)
    dataset = Dataset("a", conc, signal)
    with pytest.raises(AttributeError):
        dataset.name = "b"  # ty: ignore[invalid-assignment]
