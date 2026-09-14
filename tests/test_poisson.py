"""Grid and win-probability parity with the workbook's Poisson block."""

import numpy as np
import pytest
from scipy.stats import poisson

from backend.model.poisson import matchup, total_probabilities


def test_matchup_matches_hand_built_grid():
    home_lambda, away_lambda = 3.1, 2.6
    goals = np.arange(11)
    cells = np.outer(poisson.pmf(goals, home_lambda), poisson.pmf(goals, away_lambda))
    expected_home = (
        sum(cells[h, a] for h in goals for a in goals if h > a) + np.trace(cells) / 2
    )
    result = matchup(home_lambda, away_lambda)
    assert result["home_win_prob"] == pytest.approx(expected_home)
    assert result["home_win_prob"] + result["away_win_prob"] == pytest.approx(
        result["grid_mass"]
    )
    assert result["grid_mass"] < 1


def test_equal_lambdas_are_a_coin_flip():
    result = matchup(2.8, 2.8)
    assert result["home_win_prob"] == pytest.approx(result["away_win_prob"])


def test_total_probabilities_push_only_on_integer_lines():
    half = total_probabilities(3.0, 2.5, 6.5)
    assert half["push"] == 0
    whole = total_probabilities(3.0, 2.5, 6.0)
    assert whole["push"] > 0
    assert whole["over"] + whole["under"] + whole["push"] == pytest.approx(
        matchup(3.0, 2.5)["grid_mass"]
    )
