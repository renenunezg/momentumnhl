"""MoneyPuck team game-by-game rows: the shot-quality source behind every
window. One download per run (about 126 MB), normalized to one row per
(team, game, situation) and cached as parquet by season."""

import io
from datetime import UTC, datetime

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.etl import store

MONEYPUCK_URL = (
    "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv"
)
USER_AGENT = "Mozilla/5.0 (momentumnhl; renenunez.dev)"
REQUEST_TIMEOUT_SECONDS = 180
# The sheet's situations: even strength, power play, penalty kill. MoneyPuck's
# "all" row supplies the official goals a game finished with, for grading the
# backtest without a second source.
SITUATIONS = {"5on5": "ev", "5on4": "pp", "4on5": "pk", "all": "all"}
TEAM_CODES = {"L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL"}
COLUMNS = {
    "team": "team_abbr",
    "season": "season",
    "gameId": "game_id",
    "opposingTeam": "opponent",
    "home_or_away": "venue",
    "gameDate": "game_date",
    "situation": "situation",
    "iceTime": "toi",
    "shotsOnGoalFor": "sf",
    "goalsFor": "gf",
    "lowDangerShotsFor": "ld_sf",
    "mediumDangerShotsFor": "md_sf",
    "highDangerShotsFor": "hd_sf",
    "lowDangerGoalsFor": "ld_gf",
    "mediumDangerGoalsFor": "md_gf",
    "highDangerGoalsFor": "hd_gf",
    "shotsOnGoalAgainst": "sa",
    "goalsAgainst": "ga",
    "lowDangerShotsAgainst": "ld_sa",
    "mediumDangerShotsAgainst": "md_sa",
    "highDangerShotsAgainst": "hd_sa",
    "lowDangerGoalsAgainst": "ld_ga",
    "mediumDangerGoalsAgainst": "md_ga",
    "highDangerGoalsAgainst": "hd_ga",
    "playoffGame": "playoff",
}
COUNT_COLUMNS = [
    "toi", "sf", "gf", "ld_sf", "md_sf", "hd_sf", "ld_gf", "md_gf", "hd_gf",
    "sa", "ga", "ld_sa", "md_sa", "hd_sa", "ld_ga", "md_ga", "hd_ga",
]  # fmt: skip


def download() -> tuple[bytes, datetime]:
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=3, backoff_factor=2.0, status_forcelist=(429, 500, 502, 503, 504)
            )
        ),
    )
    response = session.get(
        MONEYPUCK_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.content, datetime.now(UTC)


def normalize(csv_bytes: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(csv_bytes), usecols=list(COLUMNS))
    frame = frame.rename(columns=COLUMNS)
    frame = frame[frame["playoff"].eq(0) & frame["situation"].isin(SITUATIONS)]
    frame = frame.drop(columns="playoff")
    frame["situation"] = frame["situation"].map(SITUATIONS)
    frame["team_abbr"] = frame["team_abbr"].replace(TEAM_CODES)
    frame["opponent"] = frame["opponent"].replace(TEAM_CODES)
    frame["venue"] = frame["venue"].str.lower()
    frame["game_id"] = frame["game_id"].astype("int64").astype(str)
    frame["game_date"] = pd.to_datetime(frame["game_date"].astype(str)).dt.date
    frame["season"] = frame["season"].astype(int)
    frame[COUNT_COLUMNS] = frame[COUNT_COLUMNS].astype(float)
    order = ["season", "game_date", "game_id", "team_abbr"]
    return frame.sort_values(order).reset_index(drop=True)


def ingest(seasons) -> dict:
    """Download once, write one parquet per requested season, return the receipt."""
    content, fetched_at = download()
    frame = normalize(content)
    wanted = set(int(s) for s in seasons)
    for season, rows in frame.groupby("season"):
        if season in wanted:
            store.write_raw(rows, "moneypuck", f"{season}.parquet")
    return store.receipt("moneypuck", content, fetched_at)


def read_games(seasons) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = store.raw_path("moneypuck", f"{season}.parquet")
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError("No MoneyPuck parquet cached; run ingest first")
    return pd.concat(frames, ignore_index=True)
