"""Independent Poisson scores with bounded tail error and optional OT/SO.

Overtime mode requires regulation-only goal rates. Every regulation tie
becomes a one-goal final margin, including the official shootout goal.
"""

import numpy as np
from scipy.stats import poisson

from backend.config import GRID_MAX_GOALS

TAIL_TOLERANCE = 1e-12


def grid(home_lambda: float, away_lambda: float, max_goals: int = GRID_MAX_GOALS):
    """Rows are home goals, columns away goals; max_goals is a minimum."""
    rates = np.asarray([home_lambda, away_lambda], dtype=float)
    if not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("Poisson rates must be finite and nonnegative")
    limit = max(max_goals, int(poisson.ppf(1 - TAIL_TOLERANCE / 2, rates).max()))
    goals = np.arange(limit + 1)
    cells = np.outer(poisson.pmf(goals, home_lambda), poisson.pmf(goals, away_lambda))
    return cells / cells.sum()


def matchup(
    home_lambda: float, away_lambda: float, overtime_home_probability: float = 0.5
) -> dict:
    if not 0 <= overtime_home_probability <= 1:
        raise ValueError("Overtime probability must be between zero and one")
    cells = grid(home_lambda, away_lambda)
    tie = float(np.trace(cells))
    home = float(np.tril(cells, -1).sum()) + tie * overtime_home_probability
    away = 1.0 - home
    return {
        "home_win_prob": home,
        "away_win_prob": away,
        "tie_mass": tie,
        "grid_mass": float(cells.sum()),
    }


def total_probabilities(
    home_lambda: float, away_lambda: float, line: float, *, overtime: bool = False
) -> dict:
    """Over, under, and push mass for a posted total on the same grid."""
    cells = grid(home_lambda, away_lambda)
    goals = np.arange(cells.shape[0])
    totals = goals[:, None] + goals[None, :]
    if overtime:
        totals = totals + np.eye(len(goals), dtype=int)
    return {
        "over": float(cells[totals > line].sum()),
        "under": float(cells[totals < line].sum()),
        "push": float(cells[totals == line].sum()),
    }
