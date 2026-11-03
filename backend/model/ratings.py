"""Team strengths from each team's recent goals and MoneyPuck xG, shrunk to
the league average. Attack and defense are pooled across venues; the league
home and away scoring rates carry home ice. Ratings keep the venue columns
the matchup formula and the site read, so both venues show one strength."""

from datetime import date

import pandas as pd

from backend.config import (
    MIN_WINDOW_GAMES,
    MODEL_VERSION,
    PRIOR_GAMES,
    WINDOW_GAMES,
    XG_WEIGHT,
)


class InsufficientHistory(ValueError):
    """No team has the minimum pregame history needed for a forecast."""


LEAGUE_KEYS = ("home_xgf", "home_xga", "away_xgf", "away_xga")
WINDOW_SUMS = ["gf", "ga", "xgf", "xga"]


def season_of(day: date) -> int:
    """Start year of the season a date belongs to; the league year turns
    over in September."""
    return day.year if day.month >= 9 else day.year - 1


def pregame_windows(rows: pd.DataFrame, as_of) -> tuple[pd.DataFrame, dict]:
    """The compact inputs a forecast reads: per team, sums over its last
    WINDOW_GAMES games before `as_of` at either venue, and the per-game league
    means of every completed season in `rows` (backend.model.signals.team_games
    output)."""
    as_of = pd.Timestamp(as_of).date()
    season = season_of(as_of)
    past = rows[rows.season < season]
    if past.empty:
        raise InsufficientHistory(f"No completed season before {as_of}")
    league = {
        "home_xgf": float(past.loc[past.venue.eq("home"), "gf"].mean()),
        "away_xgf": float(past.loc[past.venue.eq("away"), "gf"].mean()),
        **{f"league_{c}": float(past[c].mean()) for c in WINDOW_SUMS},
    }
    # The home side's goals against are the away side's goals for.
    league["home_xga"], league["away_xga"] = league["away_xgf"], league["home_xgf"]
    # Teams that played this season or last; older franchise codes drop out.
    recent = rows[(rows.game_date.dt.date < as_of) & rows.season.ge(season - 1)]
    window = rows[
        (rows.game_date.dt.date < as_of) & rows.team_abbr.isin(recent.team_abbr)
    ]
    window = (
        window.sort_values(["game_date", "game_id"])
        .groupby("team_abbr")
        .tail(WINDOW_GAMES)
    )
    windows = window.groupby("team_abbr", as_index=False)[WINDOW_SUMS].sum()
    venues = window.groupby(["team_abbr", "venue"]).size().unstack(fill_value=0)
    windows["window_games_home"] = windows.team_abbr.map(venues.get("home")).fillna(0)
    windows["window_games_away"] = windows.team_abbr.map(venues.get("away")).fillna(0)
    return windows, league


def team_ratings(rows: pd.DataFrame, as_of):
    """Returns (ratings, league) from team_games rows."""
    windows, league = pregame_windows(rows, as_of)
    return ratings_from_windows(windows, league, as_of)


def ratings_from_windows(windows: pd.DataFrame, league: dict, as_of):
    """Recompute a forecast from the compact, archived pregame input window.
    Each strength blends the goal and xG ratios to the league per-game mean,
    each shrunk by PRIOR_GAMES league-average games."""
    as_of = pd.Timestamp(as_of).date()
    if windows.empty:
        raise InsufficientHistory(f"No history before {as_of}")
    games = windows.window_games_home + windows.window_games_away

    def strength(goals: str, expected: str) -> pd.Series:
        ratios = [
            (windows[c] + PRIOR_GAMES * league[f"league_{c}"])
            / (games + PRIOR_GAMES)
            / league[f"league_{c}"]
            for c in (goals, expected)
        ]
        return (1 - XG_WEIGHT) * ratios[0] + XG_WEIGHT * ratios[1]

    attack, defense = strength("gf", "xgf"), strength("ga", "xga")
    ratings = pd.DataFrame(
        {
            "team_abbr": windows.team_abbr,
            "home_xgf": attack * league["home_xgf"],
            "home_xga": defense * league["home_xga"],
            "away_xgf": attack * league["away_xgf"],
            "away_xga": defense * league["away_xga"],
            "window_games_home": windows.window_games_home.astype(int),
            "window_games_away": windows.window_games_away.astype(int),
            "insufficient_window": games < MIN_WINDOW_GAMES,
            "home_attack": attack,
            "home_defense": defense,
            "away_attack": attack,
            "away_defense": defense,
        }
    )
    if ratings.insufficient_window.all():
        raise InsufficientHistory(f"No team has a sufficient window as of {as_of}")
    ratings["rating"] = 2 * (attack - defense)
    ratings["as_of"] = as_of
    ratings["model_version"] = MODEL_VERSION
    ratings = ratings.sort_values("rating", ascending=False).reset_index(drop=True)
    return ratings, {key: league[key] for key in LEAGUE_KEYS}
