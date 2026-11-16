"""Raw season stat lines for the player grade cards (nhl.player_season_stats).

One row per graded player with his regular-season numbers before the grade's
day, in the key order and scale of SKATER_LINE or GOALIE_LINE; a key ending
"*10" is stored as the value times ten, rounded. momentumweb decodes the array
(src/lib/hockey-season-stats.ts), so the order is a contract with it.

Every number is raw MoneyPuck game-log data summed over the season, all
situations unless the name says 5v5, and nothing here is adjusted or read by a
rating. Ice time is in minutes. A traded player's line covers every club.
"""

import numpy as np
import pandas as pd

from backend.etl import goalies as goalie_logs
from backend.players import sources

STAT_LINE_COLUMNS = ["season", "athlete_id", "position_group", "week", "stats"]


def _line(*keys: str) -> tuple[tuple[str, int], ...]:
    return tuple(
        (key.removesuffix("*10"), 10 if key.endswith("*10") else 1) for key in keys
    )


SKATER_LINE = _line(
    "games", "goals", "assists", "points", "shots", "pim", "hits", "blocks",
    "toi*10", "toi_pp*10", "toi_pk*10", "ixg*10", "faceoffs_won", "faceoffs_lost",
    "cf_5v5", "ca_5v5", "xgf_5v5*10", "xga_5v5*10", "oz_starts", "dz_starts",
)  # fmt: skip
GOALIE_LINE = _line(
    "games", "toi*10", "shots_against", "ga", "xga*10", "hd_shots", "hd_ga"
)  # fmt: skip


def _encode(totals: pd.DataFrame, line: tuple[tuple[str, int], ...]) -> pd.Series:
    scaled = np.column_stack(
        [np.rint(totals[key].to_numpy(float) * scale) for key, scale in line]
    ).astype(int)
    return pd.Series(list(map(list, scaled)), index=totals.index)


def _skater_totals(logs: pd.DataFrame) -> pd.DataFrame:
    logs = logs.assign(
        games=1,
        assists=logs.primary_assists + logs.secondary_assists,
        toi=logs.toi / 60,
        toi_pp=logs.toi_pp / 60,
        toi_pk=logs.toi_pk / 60,
    )
    totals = logs.groupby(logs.player_id.astype(str))[[k for k, _ in SKATER_LINE]]
    return totals.sum()


def _goalie_totals(logs: pd.DataFrame) -> pd.DataFrame:
    logs = logs.assign(games=1, toi=logs.toi / 60)
    return logs.groupby("goalie_id")[[k for k, _ in GOALIE_LINE]].sum()


def lines(day, season: int, week: int, graded: pd.DataFrame) -> pd.DataFrame:
    """A stat line for each player in `graded` (its grade rows), from the
    season's cached logs before `day`, which the grade run has just refreshed."""
    skaters = sources.skater_logs(season)
    keepers = goalie_logs.load([season])
    totals = pd.concat(
        [
            _encode(_skater_totals(skaters[skaters.game_date < day]), SKATER_LINE),
            _encode(_goalie_totals(keepers[keepers.game_date < day]), GOALIE_LINE),
        ]
    )
    players = graded[["athlete_id", "position_group"]].drop_duplicates()
    if players.athlete_id.duplicated().any():
        raise ValueError("A graded player is listed at two positions")
    out = players.assign(season=season, week=week, stats=players.athlete_id.map(totals))
    missing = out.stats.isna()
    if missing.any():
        raise ValueError(
            f"No season logs for graded players {list(out.athlete_id[missing])}"
        )
    return out[STAT_LINE_COLUMNS].reset_index(drop=True)
