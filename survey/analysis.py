"""Survey statistics where weights are the default, not an afterthought."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Response:
    respondent: str
    answers: dict[str, str]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("weight must be positive")


def weighted_frequencies(responses: list[Response], question: str) -> dict[str, float]:
    """Share of each answer, weighted. Sums to 1.0 (or {} if the question is absent)."""
    totals: dict[str, float] = defaultdict(float)
    for r in responses:
        if question in r.answers:
            totals[r.answers[question]] += r.weight
    grand = sum(totals.values())
    return (
        {k: round(v / grand, 4) for k, v in sorted(totals.items(), key=lambda kv: -kv[1])}
        if grand
        else {}
    )


def crosstab(responses: list[Response], row_q: str, col_q: str, *, normalise: bool = True) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in responses:
        if row_q in r.answers and col_q in r.answers:
            table[r.answers[row_q]][r.answers[col_q]] += r.weight
    if not normalise:
        return {row: dict(cols) for row, cols in table.items()}
    return {
        row: {col: value / sum(cols.values()) for col, value in cols.items()}
        for row, cols in table.items()
    }


def margin_of_error(p: float, n: int, *, confidence: float = 0.95) -> float:
    """Worst case (p=0.5) or exact proportion; finite population ignored."""
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    if n <= 0:
        raise ValueError("n must be positive")
    z = 1.96 if confidence == 0.95 else 1.6449
    return z * math.sqrt(p * (1 - p) / n)


def effective_sample_size(responses: list[Response]) -> float:
    """Kish's design effect: weighting always costs you effective sample size."""
    weights = [r.weight for r in responses]
    if not weights:
        return 0.0
    return sum(weights) ** 2 / sum(w * w for w in weights)


def chi_square_independence(table: dict[str, dict[str, float]], observed: bool = False) -> tuple[float, int]:
    """Returns (chi2, degrees of freedom) for a crosstab of weighted counts."""
    rows = list(table)
    cols = sorted({c for row in table.values() for c in row})
    if len(rows) < 2 or len(cols) < 2:
        return 0.0, 0
    row_sums = {r: sum(table[r].get(c, 0.0) for c in cols) for r in rows}
    col_sums = {c: sum(table[r].get(c, 0.0) for r in rows) for c in cols}
    total = sum(row_sums.values())
    if total == 0:
        return 0.0, 0
    chi2 = 0.0
    for r in rows:
        for c in cols:
            expected = row_sums[r] * col_sums[c] / total
            if expected:
                chi2 += (table[r].get(c, 0.0) - expected) ** 2 / expected
    return chi2, (len(rows) - 1) * (len(cols) - 1)


def crude_critical_value(df: int, confidence: float = 0.95) -> float:
    """Approximation of the chi-square critical value (Wilson-Hilferty)."""
    z = 1.6449 if confidence == 0.95 else 1.96
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3 if df else 0.0
