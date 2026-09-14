"""Official results from the NHL score feed and the fixture observation the
ledger settles against."""

import pandas as pd

from backend.etl.nhl_api import FINAL_STATES, REGULAR_SEASON

RESULT_COLUMNS = [
    "game_id",
    "season",
    "game_date",
    "start_date",
    "home_team_abbr",
    "away_team_abbr",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "last_period_type",
    "source",
    "source_fetched_at",
]
SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "start_date",
    "home_team",
    "away_team",
    "game_status",
    "completed",
    "home_goals",
    "away_goals",
    "observed_at",
]
SOURCE = "nhl api score"


def results(score_rows: list[dict], fetched_at) -> pd.DataFrame:
    """Final regular-season games only; a game without both scores or a
    finishing period is not final yet, whatever its state says."""
    rows = []
    for game in score_rows:
        if (
            game["game_type"] != REGULAR_SEASON
            or game["game_state"] not in FINAL_STATES
            or game["home_score"] is None
            or game["away_score"] is None
            or not game["last_period_type"]
        ):
            continue
        rows.append(
            {
                "game_id": game["game_id"],
                "season": game["season"],
                "game_date": game["game_date"],
                "start_date": game["start_date"],
                "home_team_abbr": game["home_abbr"],
                "away_team_abbr": game["away_abbr"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_goals": int(game["home_score"]),
                "away_goals": int(game["away_score"]),
                "last_period_type": game["last_period_type"],
                "source": SOURCE,
                "source_fetched_at": fetched_at,
            }
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def schedule_observation(
    score_rows: list[dict], finals: pd.DataFrame, fetched_at
) -> pd.DataFrame:
    """Every fixture the feed showed, final or not, with its current start
    time and state so a moved or cancelled game voids its decision."""
    final_ids = set(finals["game_id"])
    by_id = finals.set_index("game_id")
    rows = []
    for game in score_rows:
        done = game["game_id"] in final_ids
        rows.append(
            {
                "game_id": game["game_id"],
                "season": game["season"],
                "start_date": game["start_date"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "game_status": str(game["schedule_state"] or game["game_state"]).lower()
                if game["schedule_state"] not in (None, "OK")
                else str(game["game_state"]).lower(),
                "completed": done,
                "home_goals": int(by_id.at[game["game_id"], "home_goals"])
                if done
                else None,
                "away_goals": int(by_id.at[game["game_id"], "away_goals"])
                if done
                else None,
                "observed_at": fetched_at,
            }
        )
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)
