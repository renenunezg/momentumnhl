"""A skater's pregame features and the rating the frozen model makes of them.

Every row of `skater_table` describes one skater before one game, from his
earlier games only. `rate` turns the features into goals per game above an
average skater at his position: his fitted effect on expected goals at 5v5,
on the power play and on the penalty kill, each times the ice time he is
expected to get there, plus what his own shooting adds beyond expected goals.
The weights are fit by backend.players.fit and frozen in
backend/data_static/player_model.json.
"""

import json

import numpy as np
import pandas as pd

from backend.config import STATIC_DIR
from backend.players.onice import WINDOWS

MODEL_PATH = STATIC_DIR / "player_model.json"
# History lengths in games, and the league-average sample each rate is shrunk by.
SHORT, MEDIUM, LONG, CAREER = 20, 40, 160, 246
RATE_PRIOR_HOURS = 5.0
USAGE_PRIOR_GAMES = 5
GAME_SCORE_PRIOR_GAMES = 15
ZONE_PRIOR_STARTS = 30
SUMS = [
    "n", "toi", "toi_5v5", "toi_pp", "toi_pk", "ixg", "goals", "points",
    "game_score", "oz_starts", "dz_starts", "nz_starts", "fly_starts",
]  # fmt: skip
RATES = ("ixg", "goals", "points")
ON_ICE = {
    "5v5": {"off": ["5v5_off_s", "5v5_off_l"], "def": ["5v5_def_s", "5v5_def_l"]},
    "pp": {"off": ["pp_off_s", "pp_off_l"], "def": ["pp_def_s", "pp_def_l"]},
}
BOX_OFFENSE = ["ixg60", "g60", "pts60", "toi20", "pp20", "gs40", "exp"]
BOX_DEFENSE = ["toi20", "pk20", "gs40", "exp"]
DEPLOYMENT = ["zone", "fly"]
FEATURES = {
    "5v5": {
        "off": ON_ICE["5v5"]["off"] + BOX_OFFENSE + DEPLOYMENT,
        "def": ON_ICE["5v5"]["def"] + BOX_DEFENSE + DEPLOYMENT,
    },
    "pp": {
        "off": ON_ICE["pp"]["off"] + ["off5", "ixg60", "pts60", "pp20"],
        "def": ON_ICE["pp"]["def"] + ["def5", "pk20"],
    },
}
GAMES_PER_SEASON = 82


def load_model() -> dict:
    return json.loads(MODEL_PATH.read_text())


def _past(table: pd.DataFrame, window: int) -> pd.DataFrame:
    """Sums over each skater's previous `window` games, the current one excluded."""
    total = table.groupby("player_id")[SUMS].cumsum()
    before = total - table[SUMS]
    return before - total.groupby(table.player_id).shift(window + 1).fillna(0.0)


def position_means(logs: pd.DataFrame) -> dict:
    """League individual rates per hour by position, the prior for a small sample."""
    position = np.where(logs.position.eq("D"), "D", "F")
    hours = logs.toi.groupby(position).sum() / 3600
    return {
        pos: {
            c: float(logs[c].groupby(position).sum()[pos] / hours[pos]) for c in RATES
        }
        for pos in ("F", "D")
    }


def skater_table(logs: pd.DataFrame, means: dict, as_of=None) -> pd.DataFrame:
    """Pregame box-score features for every skater-game in `logs`. With
    `as_of`, each skater who has played in that date's season also gets a row
    for the date itself with no game in it: what is known of him today."""
    table = logs.copy()
    table["n"] = 1.0
    table["now"] = False
    if as_of is not None:
        latest = (
            table.sort_values(["game_date", "game_id"]).groupby("player_id").tail(1)
        )
        today = latest[latest.season.eq(latest.season.max())].copy()
        today[SUMS] = 0.0
        today["game_date"], today["game_id"], today["now"] = as_of, 0, True
        table = pd.concat([table, today], ignore_index=True)
    table["pos"] = np.where(table.position.eq("D"), "D", "F")
    table = table.sort_values(
        ["player_id", "game_date", "now", "game_id"], kind="stable"
    ).reset_index(drop=True)
    short, medium = _past(table, SHORT), _past(table, MEDIUM)
    long, career = _past(table, LONG), _past(table, CAREER)
    rate_names = {"ixg": "ixg60", "goals": "g60", "points": "pts60"}
    for column, name in rate_names.items():
        prior = table.pos.map({pos: means[pos][column] for pos in means})
        table[name] = (long[column] + RATE_PRIOR_HOURS * prior) / (
            long.toi / 3600 + RATE_PRIOR_HOURS
        )
    games = short.n + USAGE_PRIOR_GAMES
    table["toi20"] = short.toi / games / 60
    table["pp20"] = short.toi_pp / games / 60
    table["pk20"] = short.toi_pk / games / 60
    table["toi5"] = short.toi_5v5 / games / 3600
    table["gs40"] = medium.game_score / (medium.n + GAME_SCORE_PRIOR_GAMES)
    table["exp"] = long.n / LONG
    starts = long.oz_starts + long.dz_starts + long.nz_starts
    table["zone"] = (long.oz_starts - long.dz_starts) / (starts + ZONE_PRIOR_STARTS)
    table["fly"] = long.fly_starts / (starts + long.fly_starts + ZONE_PRIOR_STARTS)
    table["career_goals"], table["career_ixg"] = career.goals, career.ixg
    return table


def attach_on_ice(
    table: pd.DataFrame, players: np.ndarray, dates: np.ndarray, effects: dict
) -> pd.DataFrame:
    """Each row's on-ice effects from the latest fit dated on or before its
    game. `effects[kind][window]` is [date, offense or defense, player]."""
    day = pd.to_datetime(table.game_date).to_numpy()
    when = np.searchsorted(dates, day, side="right") - 1
    who = np.minimum(
        np.searchsorted(players, table.player_id.to_numpy()), len(players) - 1
    )
    known = (when >= 0) & (players[who] == table.player_id.to_numpy())
    out = table.copy()
    for kind, by_window in effects.items():
        for window in WINDOWS:
            for r, role in enumerate(("off", "def")):
                values = by_window[window][np.maximum(when, 0), r, who]
                out[f"{kind}_{role}_{window}"] = np.where(known, values, 0.0)
    return out


def centered(value: pd.Series, weight: pd.Series, table: pd.DataFrame) -> pd.Series:
    """Minus the ice-time-weighted mean at the position that season. Lineups
    are almost always three forwards and two defensemen, so a position's own
    level is not identified, only differences within it."""
    by = [table.season, table.pos]
    mean = (value * weight).groupby(by).transform("sum") / weight.groupby(by).transform(
        "sum"
    )
    return value - mean


def weighted(table: pd.DataFrame, weights: dict, sign: float = 1.0) -> pd.Series:
    """Sum of features times the position's weights."""
    value = pd.Series(0.0, index=table.index)
    for pos, by_feature in weights.items():
        rows = table.pos.eq(pos)
        for feature, weight in by_feature.items():
            value[rows] += sign * weight * table.loc[rows, feature]
    return value


def shooting(table: pd.DataFrame, model: dict) -> pd.Series:
    """Goals per hour of ice time a skater's finishing is expected to add
    beyond expected goals, before the fitted weight."""
    finishing = (table.career_goals - table.career_ixg) / (
        table.career_ixg + model["finishing_prior_xg"]
    )
    return finishing * table.ixg60


def rate(table: pd.DataFrame, model: dict, weight: dict) -> pd.DataFrame:
    """Adds the rating columns. `weight` gives the centering weights for 5v5,
    power-play and penalty-kill terms (columns of `table`)."""
    out = table.copy()
    w = model["weights"]
    out["off5"] = centered(weighted(out, w["5v5"]["off"]), out[weight["5v5"]], out)
    out["def5"] = centered(
        weighted(out, w["5v5"]["def"], -1.0), out[weight["5v5"]], out
    )
    out["pp_hat"] = centered(weighted(out, w["pp"]["off"]), out[weight["pp"]], out)
    out["pk_hat"] = centered(
        weighted(out, w["pp"]["def"], -1.0), out[weight["pk"]], out
    )
    out["shoot"] = shooting(out, model)
    shot = {
        kind: {pos: {"shoot": value} for pos, value in model["shooting"][kind].items()}
        for kind in ("5v5", "pp")
    }
    out["conv5"] = centered(weighted(out, shot["5v5"]), out[weight["5v5"]], out)
    out["conv_pp"] = centered(weighted(out, shot["pp"]), out[weight["pp"]], out)
    pp_hours, pk_hours = out.pp20 / 60, out.pk20 / 60
    per_game = {
        "even_strength_offense": out.off5 * out.toi5,
        "power_play": out.pp_hat * pp_hours,
        "shooting": out.conv5 * out.toi5 + out.conv_pp * pp_hours,
        "even_strength_defense": out.def5 * out.toi5,
        "penalty_kill": out.pk_hat * pk_hours,
    }
    for name, value in per_game.items():
        out[name] = value * GAMES_PER_SEASON
    out["offense"] = out.even_strength_offense + out.power_play + out.shooting
    out["defense"] = out.even_strength_defense + out.penalty_kill
    out["total"] = out.offense + out.defense
    return out
