"""The weekly players run: this season's skater and goalie grades as of a day.

Reads the frozen model, refreshes the season in progress, rebuilds the on-ice
fit for the trailing windows, and returns rows in the shape the site's player
pages read (nhl.position_traits): one row per player and trait, with a
`grade` row on the 20 to 80 scale (50 average, 10 points a standard deviation
among qualified players at the position, stopped at 20 and 80).
"""

import math
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from backend.etl import goalies as goalie_logs
from backend.model.ratings import season_of
from backend.model.signals import FRANCHISE_ALIASES
from backend.players import goalies, onice, rating, sources
from backend.players.rating import MODEL_VERSION

# Seasons of history a rating reads: on-ice fits use the trailing 730 days,
# individual rates a skater's last 246 games.
ON_ICE_SEASONS, LOG_SEASONS = 3, 4
GRADE_MEAN, GRADE_SD, GRADE_MIN, GRADE_MAX = 50.0, 10.0, 20.0, 80.0
# A skater qualifies with this many games, or half his club's while the season is young.
SKATER_GAMES, GOALIE_GAMES = 8, 4
SKATER_TRAITS = [
    "total", "offense", "defense", "even_strength_offense", "shooting",
    "power_play", "even_strength_defense", "penalty_kill",
]  # fmt: skip
GOALIE_TRAIT = "goals_saved"
POSITION_TRAIT_COLUMNS = [
    "season", "week", "as_of", "model_version", "athlete_id", "athlete_name",
    "headshot_url", "team", "position_group", "games", "trait", "value",
    "percentile", "kind", "in_grade", "uncertainty", "reliability",
    "points_above_replacement",
]  # fmt: skip


def _percentile(values: pd.Series, group: pd.Series) -> pd.Series:
    rank = values.groupby(group).rank(method="average")
    size = values.groupby(group).transform("size")
    return (100 * (rank - 1) / (size - 1)).where(size.gt(1), 100.0)


def _grade(values: pd.Series, group: pd.Series) -> pd.Series:
    mean = values.groupby(group).transform("mean")
    sd = values.groupby(group).transform("std")
    return (GRADE_MEAN + GRADE_SD * (values - mean) / sd).clip(GRADE_MIN, GRADE_MAX)


def skater_ratings(day: date, model: dict) -> pd.DataFrame:
    """One row per skater who has played this season, rated as of `day`."""
    season = season_of(day)
    logs = pd.concat(
        [
            sources.skater_logs(s, refresh=s == season)
            for s in range(season - LOG_SEASONS + 1, season + 1)
        ],
        ignore_index=True,
    )
    logs = logs[logs.game_date < day]
    parts = []
    for s in range(season - ON_ICE_SEASONS + 1, season + 1):
        season_logs = logs[logs.season.eq(s)]
        shifts = sources.shifts(s, season_logs.game_id.unique())
        shots = sources.shots(s, refresh=s == season)
        parts.append(onice.stints(shifts, shots, season_logs))
    stints = pd.concat(parts, ignore_index=True)
    stints = stints[stints.secs > 0].reset_index(drop=True)
    players = np.unique(stints[onice.HOME + onice.AWAY].to_numpy())
    players = players[players > 0]
    dates = np.array([np.datetime64(day)], dtype="datetime64[ns]")
    effects = {}
    for kind in onice.KINDS:
        blocks = onice.game_blocks(onice.attacking_rows(stints, kind), players)
        fitted, _ = onice.ridge(
            blocks, len(players), dates, [model["penalty_hours"][kind]]
        )
        effects[kind] = {w: fitted[w][:, 0] for w in onice.WINDOWS}
    table = rating.skater_table(logs, model["position_means"], as_of=day)
    games = table[~table.now & table.season.eq(season)].groupby("player_id").size()
    now = rating.attach_on_ice(table[table.now], players, dates, effects)
    now = rating.rate(now, model, {"5v5": "toi5", "pp": "pp20", "pk": "pk20"})
    now["games"] = now.player_id.map(games).astype(int)
    now["team"] = now.team.replace(FRANCHISE_ALIASES)
    club_games = logs[logs.season.eq(season)].groupby("team").game_id.nunique()
    club_games.index = club_games.index.map(lambda t: FRANCHISE_ALIASES.get(t, t))
    needed = now.team.map(club_games).map(lambda n: min(SKATER_GAMES, math.ceil(n / 2)))
    now["qualified"] = now.games >= needed
    return now


def goalie_ratings(day: date, model: dict, born: pd.Series) -> pd.DataFrame:
    season = season_of(day)
    logs = goalie_logs.refresh(range(season - LOG_SEASONS + 1, season + 1))
    logs = logs[logs.game_date < day]
    rated = goalies.rate(logs, model["goalie"], born, day)
    rated["team"] = rated.team_abbr.replace(FRANCHISE_ALIASES)
    rated["qualified"] = rated.games >= GOALIE_GAMES
    return rated


def _rows(frame: pd.DataFrame, key: str, traits: list[str]) -> pd.DataFrame:
    """Long rows for the qualified players of `frame`: the grade from the
    first trait, then each trait with its percentile at the position."""
    pool = frame[frame.qualified]
    identity = {
        "athlete_id": pool[key].astype(str),
        "moneypuck_name": pool["name"],
        "team": pool.team,
        "position_group": pool.position_group,
        "games": pool.games.astype(float),
    }
    headline = pool[traits[0]]
    measured = {
        "grade": (
            _grade(headline, pool.position_group),
            _percentile(headline, pool.position_group),
        ),
        **{t: (pool[t], _percentile(pool[t], pool.position_group)) for t in traits},
    }
    return pd.concat(
        [
            pd.DataFrame({**identity, "trait": t, "value": v, "percentile": p})
            for t, (v, p) in measured.items()
        ],
        ignore_index=True,
    )


def run(day: date) -> pd.DataFrame:
    """nhl.position_traits rows for the season `day` falls in."""
    model = rating.load_model()
    season = season_of(day)
    skaters = skater_ratings(day, model)
    known = sources.identities(sorted(set(skaters.team)))
    known["athlete_id"] = known.player_id.astype(str)
    born = known.set_index("athlete_id").birth_date
    keepers = goalie_ratings(day, model, born)
    rows = pd.concat(
        [
            _rows(
                skaters.assign(position_group=skaters.pos), "player_id", SKATER_TRAITS
            ),
            _rows(keepers.assign(position_group="G"), "goalie_id", [GOALIE_TRAIT]),
        ],
        ignore_index=True,
    )
    rows = rows.merge(
        known.drop(columns=["player_id", "birth_date"]),
        on="athlete_id",
        how="left",
        validate="many_to_one",
    )
    opening = sources.skater_logs(season).game_date.min()
    rows["athlete_name"] = rows.roster_name.fillna(rows.moneypuck_name)
    rows["season"] = season
    rows["week"] = (day - opening).days // 7 + 1
    rows["as_of"] = datetime.now(UTC)
    rows["model_version"] = MODEL_VERSION
    rows["kind"] = None
    rows["in_grade"] = True
    # Only a goalie's grade carries an uncertainty so far: how much of his own
    # record the rating keeps, on the scale the football grades use.
    kept = keepers.set_index(keepers.goalie_id.astype(str)).reliability
    graded_goalie = rows.position_group.eq("G") & rows.trait.eq("grade")
    rows["reliability"] = rows.athlete_id.map(kept).where(graded_goalie)
    rows["uncertainty"] = GRADE_SD * np.sqrt(1.0 - rows.reliability)
    rows["points_above_replacement"] = np.nan
    return rows[POSITION_TRAIT_COLUMNS]
