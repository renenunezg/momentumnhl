"""The sheet's goal grid: independent Poisson goals for each side on a 0..10
grid, ties split evenly between the teams. The grid is not renormalized,
matching the workbook; the mass above 10 goals is reported so a caller can
see what is missing."""

import numpy as np
from scipy.stats import poisson

from backend.config import GRID_MAX_GOALS


def grid(home_lambda: float, away_lambda: float, max_goals: int = GRID_MAX_GOALS):
    """Rows are home goals, columns are away goals."""
    goals = np.arange(max_goals + 1)
    return np.outer(poisson.pmf(goals, home_lambda), poisson.pmf(goals, away_lambda))


def matchup(home_lambda: float, away_lambda: float) -> dict:
    cells = grid(home_lambda, away_lambda)
    tie = float(np.trace(cells))
    home = float(np.tril(cells, -1).sum()) + tie / 2
    away = float(np.triu(cells, 1).sum()) + tie / 2
    return {
        "home_win_prob": home,
        "away_win_prob": away,
        "tie_mass": tie,
        "grid_mass": float(cells.sum()),
    }


def total_probabilities(home_lambda: float, away_lambda: float, line: float) -> dict:
    """Over, under, and push mass for a posted total on the same grid."""
    cells = grid(home_lambda, away_lambda)
    goals = np.arange(cells.shape[0])
    totals = goals[:, None] + goals[None, :]
    return {
        "over": float(cells[totals > line].sum()),
        "under": float(cells[totals < line].sum()),
        "push": float(cells[totals == line].sum()),
    }
