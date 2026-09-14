"""The sheet's Expected Goals and Model Outputs blocks: expected goals for and
against per game at each venue, strengths as ratios to the league average at
that venue, and the overall rating."""

from datetime import date

import pandas as pd

from backend.config import MODEL_VERSION
from backend.features.windows import feature_rates, venue_windows
from backend.model.goal_map import predict_per60

LEAGUE_KEYS = ("home_xgf", "home_xga", "away_xgf", "away_xga")


def season_of(day: date) -> int:
    """Start year of the season a date belongs to; the league year turns
    over in September."""
    return day.year if day.month >= 9 else day.year - 1


def _expected_goals(rates: pd.DataFrame, goal_map: dict) -> pd.DataFrame:
    """Per (team, venue): xgf = EV + PP goals for per game, xga = EV + PK
    goals against per game, each situation's per-60 scaled by its TOI share."""
    rates = rates.copy()
    share = rates["toi_per_game"] / 3600
    rates["gf_pg"] = predict_per60(rates, "for", goal_map) * share
    rates["ga_pg"] = predict_per60(rates, "against", goal_map) * share
    wide = rates.pivot(
        index=["team_abbr", "venue"],
        columns="situation",
        values=["gf_pg", "ga_pg", "games", "insufficient"],
    )
    out = pd.DataFrame(
        {
            "xgf": wide[("gf_pg", "ev")] + wide[("gf_pg", "pp")],
            "xga": wide[("ga_pg", "ev")] + wide[("ga_pg", "pk")],
            "games": wide[("games", "ev")],
            "insufficient": wide[("insufficient", "ev")].astype(bool),
        }
    ).reset_index()
    return out


def team_ratings(games: pd.DataFrame, as_of, goal_map: dict):
    """Returns (ratings, league). ratings has one row per team; league holds
    the venue averages the matchup formula multiplies by."""
    as_of = pd.Timestamp(as_of).date()
    # A 25-game venue window never needs more than the previous season, and
    # limiting the pool keeps relocated or renamed franchises out of the table.
    recent = games[games["season"] >= season_of(as_of) - 1]
    rates = feature_rates(venue_windows(recent, as_of))
    venues = _expected_goals(rates, goal_map)
    table = venues.pivot(
        index="team_abbr",
        columns="venue",
        values=["xgf", "xga", "games", "insufficient"],
    )
    ratings = pd.DataFrame(
        {
            "team_abbr": table.index,
            "home_xgf": table[("xgf", "home")].to_numpy(),
            "home_xga": table[("xga", "home")].to_numpy(),
            "away_xgf": table[("xgf", "away")].to_numpy(),
            "away_xga": table[("xga", "away")].to_numpy(),
            "window_games_home": table[("games", "home")].to_numpy(),
            "window_games_away": table[("games", "away")].to_numpy(),
        }
    )
    ratings["insufficient_window"] = (
        table[("insufficient", "home")].astype(bool).to_numpy()
        | table[("insufficient", "away")].astype(bool).to_numpy()
    )
    for column in ("home_xgf", "home_xga", "away_xgf", "away_xga"):
        ratings[column] = ratings[column].astype(float)
    ratings["window_games_home"] = ratings["window_games_home"].astype(int)
    ratings["window_games_away"] = ratings["window_games_away"].astype(int)
    usable = ratings[~ratings["insufficient_window"]]
    if usable.empty:
        raise ValueError(f"No team has a sufficient window as of {as_of}")
    league = {key: float(usable[key].mean()) for key in LEAGUE_KEYS}
    ratings["home_attack"] = ratings["home_xgf"] / league["home_xgf"]
    ratings["home_defense"] = ratings["home_xga"] / league["home_xga"]
    ratings["away_attack"] = ratings["away_xgf"] / league["away_xgf"]
    ratings["away_defense"] = ratings["away_xga"] / league["away_xga"]
    ratings["rating"] = (ratings["home_attack"] + ratings["away_attack"]) - (
        ratings["home_defense"] + ratings["away_defense"]
    )
    ratings["as_of"] = as_of
    ratings["model_version"] = MODEL_VERSION
    return ratings.sort_values("rating", ascending=False).reset_index(drop=True), league
