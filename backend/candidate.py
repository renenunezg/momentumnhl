"""Unpublished candidate forecasts from a locked chronological selection.

This path never calls publish.py. Its independently calibrated moneyline
probability and total point estimate must not be passed to v1 pick pricing.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from backend import backtest
from backend.etl import goalies as goalie_etl
from backend.etl import moneypuck, nhl_api
from backend.model import calibration, goalies, signals
from backend.model.ratings import season_of
from backend.model.signals import Specification


def project(day, selection_path: Path, confirmations: pd.DataFrame | None = None):
    selection = json.loads(selection_path.read_text())
    last_training_season = selection["holdout_season"]
    if season_of(day) <= last_training_season:
        raise ValueError("Candidate inference must be after the evaluation season")
    games = moneypuck.read_games(range(2021, season_of(day) + 1))
    games = games[games.game_date < day]
    historical = pd.concat(
        [
            backtest.official_results(games, s)
            for s in range(2022, last_training_season + 1)
        ],
        ignore_index=True,
    )
    schedule, receipt = nhl_api.schedule(day)
    future = pd.DataFrame([g for g in schedule if g["game_date"] >= day])
    if future.empty:
        return pd.DataFrame()
    forecast_at = datetime.now(UTC)
    future = future[pd.to_datetime(future.start_date, utc=True) > forecast_at].copy()
    if future.empty:
        return pd.DataFrame()
    for column in ("home_goals", "away_goals", "last_period_type"):
        future[column] = None
    fixtures = pd.concat([historical, future], ignore_index=True)
    fixtures["forecast_at"] = pd.to_datetime(fixtures.game_date, utc=True)
    fixtures.loc[fixtures.game_id.isin(future.game_id), "forecast_at"] = forecast_at
    features = signals.prepare(games, fixtures)
    needs_goalies = any(
        selection[t]["specification"]["goalie_weight"] for t in ("probability", "total")
    )
    if needs_goalies:
        history = goalie_etl.load(range(2021, last_training_season + 1))
        features = features.merge(
            goalies.forecast(history, fixtures, confirmations),
            on="game_id",
            validate="one_to_one",
        )
    result = future[
        ["game_id", "game_date", "start_date", "home_team", "away_team"]
    ].copy()
    for target, column in (("probability", "home_win_prob"), ("total", "model_total")):
        chosen = selection[target]
        predictions = signals.predict(
            features, Specification(**chosen["specification"])
        )
        training = predictions[predictions.season <= last_training_season]
        current = predictions[predictions.game_id.isin(future.game_id)].set_index(
            "game_id"
        )
        values = current[column]
        if chosen["calibrated"]:
            if target == "probability":
                fit = calibration.fit_probability(
                    training.home_win_prob,
                    (training.home_goals > training.away_goals).astype(int),
                )
                values = pd.Series(
                    calibration.probability(values, fit), index=values.index
                )
            else:
                fit = calibration.fit_total(
                    training.model_total, training.home_goals + training.away_goals
                )
                values = values + fit["offset"]
        result[column] = result.game_id.map(values)
        result[f"{target}_model"] = chosen["name"]
    result["away_win_prob"] = 1 - result.home_win_prob
    result["forecast_at"] = forecast_at
    result["model_version"] = "nhl-xg-candidate-v2"
    result["schedule_receipt_sha256"] = receipt["sha256"]
    result["status"] = "shadow_only"
    return result
