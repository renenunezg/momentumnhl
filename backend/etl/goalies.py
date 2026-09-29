"""Listed MoneyPuck goalie game logs; downloads are optional and cached.

These logs are outcomes, never evidence of a pregame starter announcement.
"""

import io
import zipfile
from datetime import UTC, datetime

import pandas as pd
import requests

from backend.etl import store
from backend.etl.moneypuck import TEAM_CODES, USER_AGENT

URL = "https://peter-tanner.com/moneypuck/downloads/seasonPlayersSummary/goalies"


def normalize(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        frames = [
            pd.read_csv(archive.open(n))
            for n in archive.namelist()
            if n.endswith(".csv") and not n.startswith("__MACOSX/")
        ]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["situation"].eq("all")].copy()
    frame = frame.rename(
        columns={
            "playerId": "goalie_id",
            "gameId": "game_id",
            "playerTeam": "team_abbr",
            "gameDate": "game_date",
            "icetime": "toi",
            "xGoals": "xga",
            "goals": "ga",
        }
    )
    frame["game_date"] = pd.to_datetime(frame.game_date.astype(str)).dt.date
    frame["game_id"] = frame.game_id.astype(str)
    frame["goalie_id"] = frame.goalie_id.astype(str)
    frame["team_abbr"] = frame.team_abbr.replace(TEAM_CODES)
    frame = frame[frame.game_id.str[4:6].eq("02")]
    return frame[
        [
            "goalie_id",
            "name",
            "season",
            "game_id",
            "game_date",
            "team_abbr",
            "toi",
            "xga",
            "ga",
        ]
    ].sort_values("game_date")


def load(seasons, *, download: bool = False) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = store.raw_path("goalies", f"{season}.parquet")
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing goalie cache for {season}")
            response = requests.get(
                f"{URL}/{season}.zip", timeout=90, headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
            frame = normalize(response.content)
            store.write_parquet(frame, path)
            store.receipt(f"goalies_{season}", response.content, datetime.now(UTC))
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)
