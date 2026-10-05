"""Probability conservation and official final-score settlement boundaries."""

import pytest
from scipy.stats import poisson, skellam

from backend.model.poisson import matchup, total_probabilities


def test_matchup_matches_hand_built_grid():
    # Opening-night Carolina rates lost 1.1% probability with the old grid.
    for home_lambda, away_lambda in (
        (4.825887, 2.831421),
        (12.0, 9.0),
        (2.8, 2.8),
        (0, 0),
    ):
        result = matchup(home_lambda, away_lambda)
        expected = (
            skellam.sf(0, home_lambda, away_lambda)
            + skellam.pmf(0, home_lambda, away_lambda) / 2
            if home_lambda and away_lambda
            else 0.5
        )
        assert result["home_win_prob"] == pytest.approx(expected, abs=1e-11)
        assert result["home_win_prob"] + result["away_win_prob"] == pytest.approx(1)
        assert result["grid_mass"] == pytest.approx(1)
    with pytest.raises(ValueError):
        matchup(float("nan"), 3)


def test_total_probabilities_push_only_on_integer_lines():
    half = total_probabilities(3.0, 2.5, 6.5)
    assert half["push"] == 0
    whole = total_probabilities(3.0, 2.5, 6.0)
    assert whole["push"] > 0
    assert whole["over"] + whole["under"] + whole["push"] == pytest.approx(1)
    # Final NHL scores cannot tie. A 3-3 regulation tie settles at seven,
    # whether decided in OT or a shootout; it cannot push a total of six.
    final = total_probabilities(3.0, 2.5, 6.0, overtime=True)
    tie_at_three = poisson.pmf(3, 3.0) * poisson.pmf(3, 2.5)
    assert final["push"] == pytest.approx(whole["push"] - tie_at_three)
    assert final["over"] == pytest.approx(whole["over"] + tie_at_three)
    assert sum(final.values()) == pytest.approx(1)
