"""Walk-forward replay: for every game date, ratings from windows that end
the day before, then the same matchup math the live run uses. Model-only,
since no free closing-line archive exists for the NHL."""

import numpy as np
import pandas as pd

from backend.config import BACKTEST_SEASONS
from backend.etl import nhl_api, store
from backend.etl.nhl_api import FINAL_STATES
from backend.model.projections import project_games
from backend.model.ratings import team_ratings

BACKTEST_COLUMNS = [
    "game_id",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "last_period_type",
    "home_lambda",
    "away_lambda",
    "home_win_prob",
    "model_total",
]


def official_results(games: pd.DataFrame, season: int) -> pd.DataFrame:
    """Final scores from the NHL API, cached per season. MoneyPuck omits the
    shootout-deciding goal, so a season's ties need the official record."""
    path = store.raw_path("results", f"{season}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    teams = sorted(games.loc[games["season"].eq(season), "team_abbr"].unique())
    rows = []
    for team in teams:
        for game in nhl_api.club_season(team, season)[0]:
            if game["home_abbr"] == team and game["game_state"] in FINAL_STATES:
                rows.append(game)
    frame = pd.DataFrame(rows).drop_duplicates("game_id")
    frame = pd.DataFrame(
        {
            "game_id": frame["game_id"],
            "season": season,
            "game_date": frame["game_date"],
            "home_abbr": frame["home_abbr"],
            "away_abbr": frame["away_abbr"],
            "home_goals": frame["home_score"].astype(int),
            "away_goals": frame["away_score"].astype(int),
            "last_period_type": frame["last_period_type"],
        }
    ).sort_values(["game_date", "game_id"])
    store.write_parquet(frame, path)
    return frame


def run(games: pd.DataFrame, goal_map: dict, seasons=BACKTEST_SEASONS) -> pd.DataFrame:
    rows = []
    for season in seasons:
        fixtures = official_results(games, season)
        for day, slate in fixtures.groupby("game_date"):
            try:
                ratings, league = team_ratings(games, day, goal_map)
            except ValueError:
                continue
            schedule = [
                {
                    "game_id": g.game_id,
                    "season": season,
                    "game_date": day,
                    "start_date": None,
                    "home_abbr": g.home_abbr,
                    "away_abbr": g.away_abbr,
                    "home_team": g.home_abbr,
                    "away_team": g.away_abbr,
                }
                for g in slate.itertuples(index=False)
            ]
            projected = project_games(schedule, ratings, league, day)
            projected = projected[projected["missing_input_count"].eq(0)]
            merged = projected.merge(
                slate[["game_id", "home_goals", "away_goals", "last_period_type"]],
                on="game_id",
            )
            rows.append(merged)
    frame = pd.concat(rows, ignore_index=True)
    return frame.reindex(columns=BACKTEST_COLUMNS)


def metrics(frame: pd.DataFrame) -> dict:
    """Probability and total accuracy summaries for a set of graded games."""
    home_won = (frame["home_goals"] > frame["away_goals"]).to_numpy(dtype=float)
    p = frame["home_win_prob"].to_numpy(dtype=float).clip(1e-6, 1 - 1e-6)
    total = (frame["home_goals"] + frame["away_goals"]).to_numpy(dtype=float)
    model_total = frame["model_total"].to_numpy(dtype=float)
    bins = np.minimum((p * 10).astype(int), 9)
    calibration = [
        {
            "bin": int(b),
            "n": int((bins == b).sum()),
            "predicted": float(p[bins == b].mean()) if (bins == b).any() else None,
            "observed": float(home_won[bins == b].mean())
            if (bins == b).any()
            else None,
        }
        for b in range(10)
    ]
    return {
        "n": int(len(frame)),
        "log_loss": float(
            -np.mean(home_won * np.log(p) + (1 - home_won) * np.log(1 - p))
        ),
        "brier": float(np.mean((p - home_won) ** 2)),
        "accuracy": float(np.mean((p > 0.5) == (home_won == 1))),
        "home_win_rate": float(home_won.mean()),
        "total_mae": float(np.mean(np.abs(model_total - total))),
        "total_bias": float(np.mean(model_total - total)),
        "calibration": calibration,
    }
