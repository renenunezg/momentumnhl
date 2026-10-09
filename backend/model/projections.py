"""The sheet's matchup rows: expected goals for each side from the two teams'
strengths and the league venue averages, then win probabilities, fair prices,
the minimum acceptable price, and the model total. A game the books quote is
then anchored to the market."""

import pandas as pd
from scipy.optimize import brentq

from backend.config import MARKET_ANCHOR_W_MODEL, MODEL_VERSION
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
    "pure_home_win_prob",
    "pure_model_total",
    "market_home_prob",
    "market_total",
]


def _priced(home_lambda: float, away_lambda: float) -> dict:
    """Everything a row derives from its two goal rates."""
    probs = matchup(home_lambda, away_lambda)
    home_decimal = 1 / probs["home_win_prob"]
    away_decimal = 1 / probs["away_win_prob"]
    return {
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
    }


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
        priced = _priced(home_lambda, away_lambda)
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
                **priced,
                "missing_input_count": int(by_team.at[home, "insufficient_window"])
                + int(by_team.at[away, "insufficient_window"]),
                "pure_home_win_prob": priced["home_win_prob"],
                "pure_model_total": priced["model_total"],
                "market_home_prob": None,
                "market_total": None,
            }
        )
    return pd.DataFrame(rows, columns=PROJECTION_COLUMNS)


def anchor_to_market(projections: pd.DataFrame, market: dict) -> pd.DataFrame:
    """Publish the market-anchored forecast for every game in `market`, which
    maps a game_id to its de-vigged `home_prob`, its posted `total`, or both.
    The win probability and the goal total each blend with their quote, then
    the two goal rates are set to return exactly that pair on the Poisson
    grid, so the prices, the picks, and the in-game anchor all read one
    forecast. A side the books do not quote keeps its pure value."""
    out = projections.copy()
    for index, row in projections.iterrows():
        quoted = market.get(row.game_id)
        if not quoted:
            continue
        target, total = row.pure_home_win_prob, row.pure_model_total
        if "home_prob" in quoted:
            target = pricing.anchored_home_probability(target, quoted["home_prob"])
        if "total" in quoted:
            total = (
                MARKET_ANCHOR_W_MODEL * total
                + (1 - MARKET_ANCHOR_W_MODEL) * quoted["total"]
            )
        share = brentq(
            lambda s: matchup(s * total, (1 - s) * total)["home_win_prob"] - target,
            1e-6,
            1 - 1e-6,
            xtol=1e-13,
        )
        anchored = _priced(share * total, (1 - share) * total)
        # The published total is the blend itself, not its float round trip.
        anchored["model_total"] = total
        anchored["market_home_prob"] = quoted.get("home_prob")
        anchored["market_total"] = quoted.get("total")
        for column, value in anchored.items():
            out.at[index, column] = value
    return out
