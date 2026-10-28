from __future__ import annotations

import pytest

from survey.analysis import (
    Response,
    chi_square_independence,
    crosstab,
    crude_critical_value,
    effective_sample_size,
    margin_of_error,
    weighted_frequencies,
)

DATA = [
    Response("r1", {"city": "Berlin", "vote": "A", "buy": "yes"}, 2.0),
    Response("r2", {"city": "Berlin", "vote": "A", "buy": "no"}, 1.0),
    Response("r3", {"city": "Paris", "vote": "B", "buy": "yes"}, 1.0),
    Response("r4", {"city": "Paris", "vote": "B", "buy": "yes"}, 3.0),
]


def test_weights_change_the_answer():
    unweighted = [Response(r.respondent, r.answers, 1.0) for r in DATA]
    assert weighted_frequencies(unweighted, "vote") == {"A": 0.5, "B": 0.5}
    assert weighted_frequencies(DATA, "vote") == {"A": 0.4286, "B": 0.5714}


def test_frequencies_sum_to_one():
    assert sum(weighted_frequencies(DATA, "city").values()) == pytest.approx(1.0)


def test_missing_question_returns_empty():
    assert weighted_frequencies(DATA, "unknown") == {}


def test_crosstab_normalises_rows():
    table = crosstab(DATA, "city", "buy")
    assert table["Berlin"]["yes"] == pytest.approx(2 / 3)
    assert table["Paris"]["yes"] == pytest.approx(1.0)


def test_crosstab_can_return_counts():
    table = crosstab(DATA, "city", "vote", normalise=False)
    assert table["Berlin"]["A"] == 3.0


def test_margin_of_error_worst_case():
    assert margin_of_error(0.5, 1000) == pytest.approx(0.031, abs=0.001)


def test_margin_of_error_shrinks_with_n():
    assert margin_of_error(0.5, 100) > margin_of_error(0.5, 10000)


def test_margin_of_error_validation():
    with pytest.raises(ValueError):
        margin_of_error(1.5, 10)
    with pytest.raises(ValueError):
        margin_of_error(0.5, 0)


def test_effective_sample_size_penalises_extreme_weights():
    equal = [Response(str(i), {}, 1.0) for i in range(10)]
    skewed = [Response(str(i), {}, 1.0) for i in range(9)] + [Response("x", {}, 100.0)]
    assert effective_sample_size(equal) == pytest.approx(10.0)
    assert effective_sample_size(skewed) < 5


def test_chi_square_detects_no_relationship_in_independent_data():
    table = {"a": {"x": 5.0, "y": 5.0}, "b": {"x": 5.0, "y": 5.0}}
    chi2, df = chi_square_independence(table)
    assert chi2 == pytest.approx(0.0) and df == 1


def test_chi_square_large_for_dependent_data():
    table = {"a": {"x": 100.0, "y": 1.0}, "b": {"x": 1.0, "y": 100.0}}
    chi2, df = chi_square_independence(table)
    assert chi2 > crude_critical_value(df)


def test_chi_square_degenerate_table():
    assert chi_square_independence({"a": {"x": 1.0}}) == (0.0, 0)


def test_weight_must_be_positive():
    with pytest.raises(ValueError):
        Response("r", {}, 0)
