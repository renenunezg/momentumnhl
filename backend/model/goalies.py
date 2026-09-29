"""Pregame goalie talent and starter uncertainty without postgame leakage."""

import numpy as np
import pandas as pd


def forecast(
    history: pd.DataFrame,
    fixtures: pd.DataFrame,
    confirmations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Mix recent goalie usage when no timestamped confirmation is available.

    Confirmations require game_id, team_abbr, goalie_id, observed_at and source.
    Only confirmations known by the forecast cutoff can replace the mixture.
    Talent uses 40 earlier appearances with a 30-game league-average prior.
    """
    history = history.copy()
    history = history.sort_values(["game_date", "game_id", "goalie_id"])
    if history.duplicated(["game_id", "goalie_id"]).any():
        raise ValueError("Duplicate goalie-game inputs")
    history["team_abbr"] = history.team_abbr.replace({"ARI": "UTA"})
    if confirmations is not None:
        confirmations = confirmations.copy()
        confirmations["observed_at"] = pd.to_datetime(
            confirmations.observed_at, utc=True
        )
    rows = []
    for day, slate in fixtures.groupby("game_date", sort=True):
        past = history[history.game_date < day]
        last = past.groupby("goalie_id").tail(40)
        league_toi = past.toi.sum()
        league = (
            (past.xga.sum() - past.ga.sum()) * 3600 / league_toi if league_toi else 0
        )
        talent = last.groupby("goalie_id")[["xga", "ga", "toi"]].sum()
        talent["gsax60"] = ((talent.xga - talent.ga) * 3600 - league * talent.toi) / (
            talent.toi + 30 * 3600
        )
        for game in slate.itertuples(index=False):
            out = {"game_id": game.game_id}
            cutoff = pd.Timestamp(game.forecast_at)
            for side in ("home", "away"):
                team = getattr(game, f"{side}_abbr")
                team = "UTA" if team == "ARI" else team
                recent = past[past.team_abbr.eq(team)]
                game_ids = recent.drop_duplicates("game_id").tail(10).game_id
                usage = (
                    recent[recent.game_id.isin(game_ids)].groupby("goalie_id").toi.sum()
                )
                weights = usage / usage.sum() if usage.sum() else usage
                confirmed = False
                if confirmations is not None and not confirmations.empty:
                    known = confirmations[
                        confirmations.game_id.eq(game.game_id)
                        & confirmations.team_abbr.replace({"ARI": "UTA"}).eq(team)
                        & (confirmations.observed_at <= cutoff)
                        & confirmations.source.fillna("").str.len().gt(0)
                    ]
                    if len(known):
                        goalie = known.sort_values("observed_at").iloc[-1].goalie_id
                        weights = pd.Series({str(goalie): 1.0})
                        confirmed = True
                skill = talent.gsax60.reindex(weights.index).fillna(0)
                out[f"{side}_goalie_adjustment"] = float(np.dot(skill, weights))
                out[f"{side}_goalie_confirmed"] = confirmed
                out[f"{side}_goalie_probability"] = (
                    float(weights.max()) if len(weights) else 0
                )
            rows.append(out)
    return pd.DataFrame(rows)
