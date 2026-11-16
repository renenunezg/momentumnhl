"""Player inputs, cached one parquet per season under backend/data/raw.

MoneyPuck supplies skater game logs and the shot file (each shot's time and
expected-goal value). The NHL supplies who was on the ice: its shift-chart
API, and for the games that API has no rows for (it has had week-long holes)
the HTML time-on-ice reports, which carry the same shifts by sweater number.
Completed seasons never change, so only the season in progress is refreshed.
"""

import io
import re
import time
import zipfile

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.etl import store
from backend.etl.moneypuck import TEAM_CODES
from backend.etl.nhl_api import BASE_URL, USER_AGENT

MONEYPUCK = "https://peter-tanner.com/moneypuck/downloads"
SHIFT_CHARTS = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={}"
TOI_REPORT = "https://www.nhl.com/scores/htmlreports/{}{}/T{}{}.HTM"
BIOS = "https://moneypuck.com/moneypuck/playerData/playerBios/allPlayersLookup.csv"
REQUEST_PAUSE_SECONDS = 0.25
SHIFT_TYPE = 517
REPORT_PERIODS = {"1": 1, "2": 2, "3": 3, "OT": 4}
SITUATIONS = {"all": "", "5on5": "_5v5", "5on4": "_pp", "4on5": "_pk"}
ZONE_STARTS = {
    "I_F_oZoneShiftStarts": "oz_starts",
    "I_F_dZoneShiftStarts": "dz_starts",
    "I_F_neutralZoneShiftStarts": "nz_starts",
    "I_F_flyShiftStarts": "fly_starts",
}
# Box-score counts for the season stat lines (backend/players/season_stats.py),
# over all situations; no rating reads them.
BOX = {
    "I_F_primaryAssists": "primary_assists",
    "I_F_secondaryAssists": "secondary_assists",
    "I_F_shotsOnGoal": "shots",
    "I_F_penalityMinutes": "pim",
    "I_F_hits": "hits",
    "shotsBlockedByPlayer": "blocks",
    "faceoffsWon": "faceoffs_won",
    "faceoffsLost": "faceoffs_lost",
}
# On-ice shot attempts and expected goals for and against at 5v5.
ON_ICE_5V5 = {
    "OnIce_F_shotAttempts": "cf_5v5",
    "OnIce_A_shotAttempts": "ca_5v5",
    "OnIce_F_xGoals": "xgf_5v5",
    "OnIce_A_xGoals": "xga_5v5",
}
SKATER_COLUMNS = [
    "player_id", "name", "season", "game_id", "game_date", "team", "position",
    "toi", "toi_5v5", "toi_pp", "toi_pk", "game_score", "ixg", "goals", "points",
    *ZONE_STARTS.values(), *BOX.values(), *ON_ICE_5V5.values(),
]  # fmt: skip
SHOT_COLUMNS = ["season", "game_id", "time", "home", "xg", "goal", "home_team"]
SHIFT_COLUMNS = ["game_id", "player_id", "team", "period", "start", "end"]

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=4, backoff_factor=2.0, status_forcelist=(429, 500, 502, 503, 504)
        )
    ),
)


def _get(url: str, timeout: int = 300) -> requests.Response:
    response = _session.get(url, timeout=timeout)
    response.raise_for_status()
    return response


def _cached(kind: str, season: int, refresh: bool, build) -> pd.DataFrame:
    path = store.raw_path(kind, f"{season}.parquet")
    if refresh or not path.exists():
        store.write_parquet(build(season), path)
    return pd.read_parquet(path)


def _skater_logs(season: int) -> pd.DataFrame:
    """One row per skater and game from MoneyPuck's per-player season files."""
    content = _get(f"{MONEYPUCK}/seasonPlayersSummary/skaters/{season}.zip").content
    wanted = [
        "playerId", "name", "season", "gameId", "playerTeam", "gameDate", "position",
        "situation", "icetime", "gameScore", "I_F_xGoals", "I_F_goals", "I_F_points",
        *ZONE_STARTS, *BOX, *ON_ICE_5V5,
    ]  # fmt: skip
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        frames = [
            pd.read_csv(archive.open(name), usecols=wanted)
            for name in archive.namelist()
            if name.endswith(".csv") and not name.startswith("__MACOSX/")
        ]
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw.situation.isin(SITUATIONS)]
    raw = raw[raw.gameId.astype(str).str[4:6].eq("02")]
    key = ["playerId", "gameId"]
    whole = raw[raw.situation.eq("all")].rename(
        columns={
            "playerId": "player_id",
            "gameId": "game_id",
            "playerTeam": "team",
            "gameDate": "game_date",
            "icetime": "toi",
            "gameScore": "game_score",
            "I_F_xGoals": "ixg",
            "I_F_goals": "goals",
            "I_F_points": "points",
            **BOX,
        }
    )
    out = whole.drop(columns=["situation", *ZONE_STARTS, *ON_ICE_5V5])
    for situation, suffix in SITUATIONS.items():
        if not suffix:
            continue
        part = raw.loc[raw.situation.eq(situation), [*key, "icetime"]]
        part.columns = ["player_id", "game_id", f"toi{suffix}"]
        out = out.merge(
            part, on=["player_id", "game_id"], how="left", validate="one_to_one"
        )
    even = raw.loc[raw.situation.eq("5on5"), [*key, *ZONE_STARTS, *ON_ICE_5V5]]
    even = even.rename(
        columns={
            "playerId": "player_id",
            "gameId": "game_id",
            **ZONE_STARTS,
            **ON_ICE_5V5,
        }
    )
    out = out.merge(
        even, on=["player_id", "game_id"], how="left", validate="one_to_one"
    )
    filled = [
        "toi_5v5", "toi_pp", "toi_pk", *ZONE_STARTS.values(), *ON_ICE_5V5.values()
    ]  # fmt: skip
    out[filled] = out[filled].fillna(0.0)
    out["team"] = out.team.replace(TEAM_CODES)
    out["game_date"] = pd.to_datetime(out.game_date.astype(str)).dt.date
    out[["player_id", "game_id", "season"]] = out[
        ["player_id", "game_id", "season"]
    ].astype("int64")
    order = ["game_date", "game_id", "player_id"]
    return out.sort_values(order)[SKATER_COLUMNS].reset_index(drop=True)


def _shots(season: int) -> pd.DataFrame:
    """Every unblocked regular-season shot with its game clock and xG."""
    content = _get(f"{MONEYPUCK}/shots_{season}.zip").content
    wanted = [
        "season", "game_id", "isPlayoffGame", "time", "team", "homeTeamCode",
        "goal", "xGoal",
    ]  # fmt: skip
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".csv"))
        raw = pd.read_csv(archive.open(name), usecols=wanted)
    raw = raw[raw.isPlayoffGame.eq(0)]
    out = pd.DataFrame(
        {
            "season": raw.season.astype("int64"),
            "game_id": raw.season.astype("int64") * 1_000_000
            + raw.game_id.astype("int64"),
            "time": raw.time.astype(float),
            "home": raw.team.eq("HOME"),
            "xg": raw.xGoal.astype(float),
            "goal": raw.goal.astype(int),
            "home_team": raw.homeTeamCode.replace(TEAM_CODES),
        }
    )
    return out.sort_values(["game_id", "time"]).reset_index(drop=True)


def skater_logs(season: int, refresh: bool = False) -> pd.DataFrame:
    return _cached("skater_logs", season, refresh, _skater_logs)


def shots(season: int, refresh: bool = False) -> pd.DataFrame:
    return _cached("shot_logs", season, refresh, _shots)


def _clock(value: str) -> int:
    minutes, seconds = value.split(":")
    return int(minutes) * 60 + int(seconds)


def _api_shifts(game_id: int) -> list[tuple]:
    rows = _get(SHIFT_CHARTS.format(game_id), timeout=60).json()["data"]
    return [
        (
            game_id,
            row["playerId"],
            row["teamAbbrev"],
            row["period"],
            _clock(row["startTime"]),
            _clock(row["endTime"]),
        )
        for row in rows
        if row["typeCode"] == SHIFT_TYPE and row["startTime"] and row["endTime"]
    ]


def report_shifts(html: str) -> list[tuple]:
    """(sweater number, period, start, end) for every shift in one club's
    time-on-ice report. Each player's table ends with per-period totals, which
    have a different shape and are skipped."""
    out = []
    for block in re.split(r'class="playerHeading[^"]*"[^>]*>', html)[1:]:
        number = int(block.split("<", 1)[0].split()[0])
        rows = re.findall(
            r'<tr class="\s*(?:oddColor|evenColor)">(.*?)</tr>', block, flags=re.S
        )
        for row in rows:
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S)
            if len(cells) != 6 or "/" not in cells[2]:
                continue
            period = REPORT_PERIODS.get(cells[1].strip())
            if period is not None:
                start, end = (_clock(c.split("/")[0].strip()) for c in cells[2:4])
                out.append((number, period, start, end))
    return out


def _report_game(game_id: int) -> list[tuple]:
    season = int(str(game_id)[:4])
    box = _get(f"{BASE_URL}/gamecenter/{game_id}/boxscore", timeout=60).json()
    rows = []
    for side, letter in (("homeTeam", "H"), ("awayTeam", "V")):
        numbers = {
            player["sweaterNumber"]: player["playerId"]
            for group in box["playerByGameStats"][side].values()
            for player in group
        }
        url = TOI_REPORT.format(season, season + 1, letter, str(game_id)[4:])
        for number, period, start, end in report_shifts(_get(url, timeout=60).text):
            rows.append(
                (game_id, numbers[number], box[side]["abbrev"], period, start, end)
            )
        time.sleep(REQUEST_PAUSE_SECONDS)
    return rows


def shifts(season: int, game_ids) -> pd.DataFrame:
    """Shifts for `game_ids`, fetching only the games not yet cached. A game
    without two teams of shifts in either source is an error, not a gap."""
    path = store.raw_path("shifts", f"{season}.parquet")
    have = (
        pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=SHIFT_COLUMNS)
    )
    missing = sorted(set(map(int, game_ids)) - set(have.game_id.unique()))
    rows = []
    for game_id in missing:
        # The API has returned nothing for some games and another game's rows
        # mixed into others; either way the club reports are the fallback.
        found = _api_shifts(game_id)
        if len({row[2] for row in found}) != 2:
            found = _report_game(game_id)
        if len({row[2] for row in found}) != 2:
            raise ValueError(f"No usable shift data for game {game_id}")
        rows += found
        time.sleep(REQUEST_PAUSE_SECONDS)
    if rows:
        new = pd.DataFrame(rows, columns=SHIFT_COLUMNS)
        have = pd.concat([have, new], ignore_index=True)
        store.write_parquet(have, path)
    return have[have.game_id.isin(set(map(int, game_ids)))]


def identities(teams) -> pd.DataFrame:
    """Names, headshots and birth dates from each club's current roster.
    MoneyPuck's names lose their accented letters, so the site reads these
    where they exist."""
    rows = []
    for team in teams:
        payload = _get(f"{BASE_URL}/roster/{team}/current", timeout=60).json()
        for group in ("forwards", "defensemen", "goalies"):
            for player in payload[group]:
                first = player["firstName"]["default"]
                last = player["lastName"]["default"]
                rows.append(
                    {
                        "player_id": int(player["id"]),
                        "roster_name": f"{first} {last}",
                        "headshot_url": player.get("headshot"),
                        "birth_date": player.get("birthDate"),
                    }
                )
        time.sleep(REQUEST_PAUSE_SECONDS)
    return pd.DataFrame(rows).drop_duplicates("player_id")


def birth_dates() -> pd.Series:
    """Birth date by player id (as text) from MoneyPuck's player list, for
    fitting the goalie age curve on seasons whose rosters are long gone."""
    path = store.raw_path("player_bios.parquet")
    if not path.exists():
        bios = pd.read_csv(io.BytesIO(_get(BIOS).content))
        store.write_parquet(bios[["playerId", "birthDate"]], path)
    bios = pd.read_parquet(path, columns=["playerId", "birthDate"])
    born = pd.to_datetime(bios.birthDate, errors="coerce")
    return pd.Series(born.to_numpy(), index=bios.playerId.astype(str)).dropna()
