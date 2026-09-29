"""The sheet's matchup rows: expected goals for each side from the two teams'
strengths and the league venue averages, then win probabilities, fair prices,
the minimum acceptable price, and the model total."""

import pandas as pd

from backend.config import MODEL_VERSION
from backend.model import pricing
from backend.model.poisson import matchup

PROJECTION_COLUMNS = [
    "game_id",
    "season",
    "game_date",
    "start_date",
    "as_of",
    "model_version",
    "home_team_abbr",
    "away_team_abbr",
    "home_team",
    "away_team",
    "home_lambda",
    "away_lambda",
    "home_win_prob",
    "away_win_prob",
    "model_total",
    "home_fair_decimal",
    "away_fair_decimal",
    "home_fair_price",
    "away_fair_price",
    "home_minimum_price",
    "away_minimum_price",
    "missing_input_count",
]


def project_games(
    schedule_rows: list[dict], ratings: pd.DataFrame, league: dict, as_of
) -> pd.DataFrame:
    """One row per scheduled game whose teams both have ratings. Teams with
    an insufficient window still get lambdas (their short window is what the
    sheet would have used) but the row's missing_input_count blocks pricing."""
    by_team = ratings.set_index("team_abbr")
    cutoff = pd.to_datetime(as_of, utc=True)
    rows = []
    for game in schedule_rows:
        # Intraday retries may include games already underway. Their stored
        # forecast and decision must remain the pregame versions.
        start = pd.to_datetime(game["start_date"], utc=True)
        if (pd.notna(start) and start <= cutoff) or game.get("game_state") not in (
            None, "FUT", "PRE"
        ):
            continue
        home, away = game["home_abbr"], game["away_abbr"]
        if home not in by_team.index or away not in by_team.index:
            continue
        home_lambda = float(
            by_team.at[home, "home_attack"]
            * by_team.at[away, "away_defense"]
            * league["home_xgf"]
        )
        away_lambda = float(
            by_team.at[away, "away_attack"]
            * by_team.at[home, "home_defense"]
            * league["away_xgf"]
        )
        probs = matchup(home_lambda, away_lambda)
        home_decimal = 1 / probs["home_win_prob"]
        away_decimal = 1 / probs["away_win_prob"]
        rows.append(
            {
                "game_id": game["game_id"],
                "season": game["season"],
                "game_date": game["game_date"],
                "start_date": game["start_date"],
                "as_of": as_of,
                "model_version": MODEL_VERSION,
                "home_team_abbr": home,
                "away_team_abbr": away,
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_lambda": home_lambda,
                "away_lambda": away_lambda,
                "home_win_prob": probs["home_win_prob"],
                "away_win_prob": probs["away_win_prob"],
                "model_total": home_lambda + away_lambda,
                "home_fair_decimal": home_decimal,
                "away_fair_decimal": away_decimal,
                "home_fair_price": pricing.decimal_to_american(home_decimal),
                "away_fair_price": pricing.decimal_to_american(away_decimal),
                "home_minimum_price": pricing.minimum_price(home_decimal),
                "away_minimum_price": pricing.minimum_price(away_decimal),
                "missing_input_count": int(by_team.at[home, "insufficient_window"])
                + int(by_team.at[away, "insufficient_window"]),
            }
        )
    return pd.DataFrame(rows, columns=PROJECTION_COLUMNS)
