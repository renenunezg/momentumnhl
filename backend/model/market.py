"""Timestamp-gated market comparisons kept separate from pure forecasts."""

import numpy as np
import pandas as pd

from backend.model.pricing import american_to_decimal


def paired_quotes(
    predictions: pd.DataFrame, snapshots: pd.DataFrame, *, timing: str = "decision"
) -> pd.DataFrame:
    """One same-provider, same-fetch two-sided quote per game.

    Decision quotes must exist by forecast_at. Closing quotes are diagnostic
    only and must never be used to revise a morning forecast or its grade.
    """
    if timing not in ("decision", "closing"):
        raise ValueError("Unknown quote timing")
    if predictions.game_id.duplicated().any():
        raise ValueError("Predictions must contain one row per game")
    if snapshots.empty:
        return predictions.iloc[:0].copy()
    rows = predictions.merge(snapshots, on="game_id", validate="one_to_many")
    for column in ("start_date", "forecast_at", "fetched_at", "provider_last_update"):
        rows[column] = pd.to_datetime(rows[column], utc=True)
    cutoff = rows.forecast_at if timing == "decision" else rows.start_date
    valid = (
        (rows.forecast_at < rows.start_date)
        & (rows.fetched_at <= cutoff)
        & (rows.fetched_at < rows.start_date)
        & (rows.provider_last_update <= rows.fetched_at)
        & ((cutoff - rows.provider_last_update) <= pd.Timedelta(24, unit="h"))
        & rows.provider_key.isin(["draftkings", "fanduel"])
    )
    for column in ("home_price", "away_price", "over_price", "under_price"):
        valid &= np.isfinite(rows[column]) & rows[column].abs().ge(100)
    valid &= np.isfinite(rows.total_line) & rows.total_line.gt(0)
    rows = rows[valid].sort_values(["fetched_at", "provider_key"])
    rows = rows.drop_duplicates("game_id", keep="last").copy()
    h = 1 / rows.home_price.map(american_to_decimal)
    a = 1 / rows.away_price.map(american_to_decimal)
    rows["market_home_probability"] = h / (h + a)
    rows["quote_timing"] = timing
    return rows


def evaluate(
    predictions: pd.DataFrame,
    snapshots: pd.DataFrame,
    development_seasons=(2023, 2024),
    holdout_season=2025,
) -> dict:
    if not development_seasons or max(development_seasons) >= holdout_season:
        raise ValueError("Market development must precede the holdout season")
    rows = paired_quotes(predictions, snapshots)
    if rows.empty:
        return {
            "status": "blocked",
            "reason": "No eligible timestamped paired quotes",
            "eligible_games": 0,
        }
    graded = rows.dropna(
        subset=["home_goals", "away_goals", "model_total", "home_win_prob"]
    )
    train = graded[graded.season.isin(development_seasons)]
    test = graded[graded.season.eq(holdout_season)]
    coverage = {
        "eligible_games": len(rows),
        "development_games": len(train),
        "holdout_games": len(test),
    }
    if len(train) < 200 or len(test) < 100:
        return {
            "status": "blocked",
            "reason": "Insufficient chronological market coverage",
            **coverage,
        }
    comparisons = []
    train_y = (train.home_goals > train.away_goals).astype(float)
    test_y = (test.home_goals > test.away_goals).astype(float)

    def loss(probability, outcomes):
        p = probability.clip(1e-6, 1 - 1e-6)
        return float(-(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)).mean())

    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        total = (1 - weight) * train.model_total + weight * train.total_line
        mae = (total - train.home_goals - train.away_goals).abs().mean()
        probability = (
            1 - weight
        ) * train.home_win_prob + weight * train.market_home_probability
        comparisons.append(
            {
                "weight": weight,
                "development_mae": float(mae),
                "development_log_loss": loss(probability, train_y),
            }
        )
    chosen = min(comparisons, key=lambda r: (r["development_mae"], r["weight"]))
    probability_choice = min(
        comparisons, key=lambda r: (r["development_log_loss"], r["weight"])
    )
    p_weight = probability_choice["weight"]
    blended_probability = (
        1 - p_weight
    ) * test.home_win_prob + p_weight * test.market_home_probability
    actual = test.home_goals + test.away_goals
    weight = chosen["weight"]
    blend = (1 - weight) * test.model_total + weight * test.total_line
    return {
        "status": "evaluated",
        **coverage,
        "selected": chosen,
        "pure_holdout_mae": float((test.model_total - actual).abs().mean()),
        "market_holdout_mae": float((test.total_line - actual).abs().mean()),
        "blend_holdout_mae": float((blend - actual).abs().mean()),
        "probability_selected": probability_choice,
        "pure_holdout_log_loss": loss(test.home_win_prob, test_y),
        "market_holdout_log_loss": loss(test.market_home_probability, test_y),
        "blend_holdout_log_loss": loss(blended_probability, test_y),
        "development": comparisons,
    }
