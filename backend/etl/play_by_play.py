"""Historical play-by-play for the in-game backtest: one NHL API call per
game, downloaded once and cached per season. Only the fields that describe
score, clock, and manpower are kept."""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from backend.etl import nhl_api, store

COLUMNS = [
    "game_id",
    "season",
    "event_index",
    "period",
    "period_type",
    "period_seconds",
    "event_type",
    "situation_code",
    "owner_is_home",
    "home_score",
    "away_score",
    "penalty_minutes",
    "penalty_type",
]
CHECKPOINT_GAMES = 200
WORKERS = 4


def _events(game_id: str, season: int) -> list[dict]:
    payload = nhl_api.play_by_play(game_id)
    home_id = payload["homeTeam"]["id"]
    rows = []
    for index, play in enumerate(payload["plays"]):
        details = play.get("details") or {}
        minutes, seconds = (int(part) for part in play["timeInPeriod"].split(":"))
        owner = details.get("eventOwnerTeamId")
        rows.append(
            {
                "game_id": game_id,
                "season": season,
                "event_index": index,
                "period": int(play["periodDescriptor"]["number"]),
                "period_type": play["periodDescriptor"]["periodType"],
                "period_seconds": minutes * 60 + seconds,
                "event_type": play["typeDescKey"],
                "situation_code": play.get("situationCode"),
                "owner_is_home": None if owner is None else owner == home_id,
                "home_score": details.get("homeScore"),
                "away_score": details.get("awayScore"),
                "penalty_minutes": details.get("duration"),
                "penalty_type": details.get("typeCode"),
            }
        )
    if not rows:
        raise ValueError(f"Empty play-by-play for game {game_id}")
    return rows


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["owner_is_home"] = frame["owner_is_home"].astype("boolean")
    for column in ("home_score", "away_score", "penalty_minutes"):
        frame[column] = frame[column].astype("Int64")
    return frame


def ingest(games: pd.DataFrame) -> dict:
    """Fetch every listed game not already cached. `games` needs game_id and
    season. Progress is saved every few hundred games, so a rerun resumes."""
    summary = {}
    for season, slate in games.groupby("season"):
        path = store.raw_path("play_by_play", f"{season}.parquet")
        cached = pd.read_parquet(path) if path.exists() else _frame([])
        missing = sorted(set(slate["game_id"].astype(str)) - set(cached["game_id"]))
        for start in range(0, len(missing), CHECKPOINT_GAMES):
            batch = missing[start : start + CHECKPOINT_GAMES]
            with ThreadPoolExecutor(WORKERS) as pool:
                fetched = list(pool.map(lambda g: _events(g, int(season)), batch))
            new = _frame([row for rows in fetched for row in rows])
            cached = (
                new if cached.empty else pd.concat([cached, new], ignore_index=True)
            )
            store.write_parquet(cached, path)
            print(
                f"{season}: {start + len(batch)} of {len(missing)} fetched", flush=True
            )
        summary[int(season)] = {
            "games": int(cached["game_id"].nunique()),
            "fetched": len(missing),
        }
    return summary


def read(seasons) -> pd.DataFrame:
    return pd.concat(
        [store.read_raw("play_by_play", f"{season}.parquet") for season in seasons],
        ignore_index=True,
    )
