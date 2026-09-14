"""Odds arithmetic checked against the numbers the workbook displays."""

import pytest

from backend.model.pricing import (
    american_to_decimal,
    decimal_to_american,
    devig,
    expected_value_per_unit,
    implied_probability,
    kelly_fraction,
    minimum_price,
    probability_edge,
)


def test_sheet_examples():
    # Model Outputs: -130 shows 1.77 and 56.5%, +110 shows 2.10 and 47.6%.
    assert american_to_decimal(-130) == pytest.approx(1.769, abs=1e-3)
    assert implied_probability(-130) == pytest.approx(0.565, abs=1e-3)
    assert american_to_decimal(110) == pytest.approx(2.10)
    assert decimal_to_american(2.10) == 110
    assert decimal_to_american(1.77) == -130
    # 4.1% juice on that pair.
    assert implied_probability(-130) + implied_probability(110) - 1 == pytest.approx(
        0.041, abs=1e-3
    )
    assert sum(devig(0.565, 0.476)) == pytest.approx(1.0)


def test_kelly_ev_edge_and_minimum_price():
    assert kelly_fraction(0.60, 1.77, 0.10) == pytest.approx(
        0.10 * (0.60 * 1.77 - 1) / 0.77
    )
    assert kelly_fraction(0.40, 1.77, 0.10) == 0
    # Sheet EV per unit: p * (d - 1) - (1 - p).
    assert expected_value_per_unit(0.60, -130) == pytest.approx(
        0.60 * 0.769 - 0.40, abs=1e-3
    )
    # The sheet's play rule compares model probability with the book's
    # implied probability; that difference is the edge over break-even.
    assert probability_edge(0.60, -130) == pytest.approx(0.60 - 0.565, abs=1e-3)
    assert minimum_price(1.77) == decimal_to_american(1.77 * 1.1)
