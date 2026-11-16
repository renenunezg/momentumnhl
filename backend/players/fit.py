"""Fit the frozen player model: backend/data_static/player_model.json.

Run by hand when the method or the seasons change, never by the weekly job.
The expected goals of every stint in the fit seasons are regressed on the
summed pregame features of the skaters on the ice, forwards and defensemen
weighted separately, so a skater's rating is the part of his line's results
his own history predicts. Power-play stints are fit the same way, and actual
goals are then regressed on the expected-goal forecast plus the attackers'
own shooting. Every feature of a game comes from earlier games only.
"""

import json

import numpy as np
import pandas as pd

from backend.config import HISTORY_START_SEASON
from backend.etl import goalies as goalie_logs
from backend.players import goalies, onice, rating, sources

STEP_DAYS = 14
PENALTIES = {"5v5": (1.0, 2.0, 4.0, 8.0, 16.0, 32.0), "pp": (4.0, 8.0, 16.0, 32.0)}
FINISHING_PRIORS = (10.0, 20.0, 40.0, 80.0, 160.0)
FUTURE_GAMES, MIN_FUTURE_GAMES = 40, 20
KEY = 10_000_000


class Design:
    """Summed features of the forwards and of the defensemen on each side of
    every attacking row of one kind of stint."""

    def __init__(self, stints, table, kind, features):
        self.rows = rows = onice.attacking_rows(stints, kind)
        self.features = features
        self.season = stints.season.to_numpy()[rows["stint"]]
        played = table[~table.now]
        key = played.game_id.to_numpy() * KEY + played.player_id.to_numpy()
        order = np.argsort(key)
        key = key[order]
        values = played[features].to_numpy(np.float32)[order]
        forward = played.pos.eq("F").to_numpy()[order]
        self.sums = {}
        for role in ("off", "def"):
            ids = rows[role]
            want = rows["game"][:, None] * KEY + ids
            at = np.minimum(np.searchsorted(key, want), len(key) - 1)
            found = (key[at] == want) & (ids > 0)
            for pos, flag in (("F", forward[at]), ("D", ~forward[at])):
                total = np.zeros((len(want), len(features)), np.float32)
                for slot in range(ids.shape[1]):
                    use = (found[:, slot] & flag[:, slot])[:, None]
                    total += values[at[:, slot]] * use
                self.sums[role, pos] = total

    def matrix(self, offense, defense, extra=None):
        column = {f: i for i, f in enumerate(self.features)}
        parts, names = [], []
        for role, wanted in (("off", offense), ("def", defense)):
            for pos in ("F", "D"):
                parts.append(self.sums[role, pos][:, [column[f] for f in wanted]])
                names += [(role, pos, f) for f in wanted]
        parts.append(self.rows["nuisance"])
        names += [("nuisance", "", i) for i in range(self.rows["nuisance"].shape[1])]
        if extra is not None:
            parts.append(extra)
            names += [("extra", "", i) for i in range(extra.shape[1])]
        return np.hstack(parts).astype(np.float64), names

    def fit(self, offense, defense, target, train, extra=None):
        """Weighted least squares of the target per hour, weights hours, on the
        rows flagged `train`. Returns predictions for every row, the weights
        by role and position, and the training error."""
        X, names = self.matrix(offense, defense, extra)
        hours, total = self.rows["hours"], self.rows[target]
        gram = (X[train] * hours[train, None]).T @ X[train]
        beta = np.linalg.lstsq(gram, X[train].T @ total[train], rcond=None)[0]
        predicted = X @ beta
        weights = {}
        for (role, pos, feature), value in zip(names, beta):
            if role in ("off", "def"):
                weights.setdefault(role, {}).setdefault(pos, {})[feature] = float(value)
        error = (total[train] - predicted[train] * hours[train]) ** 2 / hours[train]
        return predicted, weights, float(error.sum() / hours[train].sum())


def history(seasons):
    """Skater logs and stints for `seasons`, from the caches."""
    logs = pd.concat([sources.skater_logs(s) for s in seasons], ignore_index=True)
    parts = []
    for season in seasons:
        season_logs = logs[logs.season.eq(season)]
        shifts = sources.shifts(season, season_logs.game_id.unique())
        part = onice.stints(shifts, sources.shots(season), season_logs)
        parts.append(part.assign(season=season))
    stints = pd.concat(parts, ignore_index=True)
    return logs, stints[stints.secs > 0].reset_index(drop=True)


def skaters_on_ice(stints) -> np.ndarray:
    players = np.unique(stints[onice.HOME + onice.AWAY].to_numpy())
    return players[players > 0]


def _future(table, columns):
    """Sums over the current and next games of the same season, up to FUTURE_GAMES."""
    group = table.groupby(["player_id", "season"])
    total = group[columns].cumsum()
    end = total.groupby([table.player_id, table.season]).shift(-(FUTURE_GAMES - 1))
    whole = group[columns].transform("sum")
    return end.where(end.n.notna(), whole) - (total - table[columns])


def fit(seasons) -> dict:
    """The model fit on stints of `seasons`; earlier seasons back to
    HISTORY_START_SEASON only supply history."""
    seasons = sorted(seasons)
    logs, stints = history(range(HISTORY_START_SEASON, seasons[-1] + 1))
    players = skaters_on_ice(stints)
    dates = pd.date_range(
        stints.day.min() + pd.Timedelta(days=STEP_DAYS),
        stints.day.max() + pd.Timedelta(days=STEP_DAYS),
        freq=f"{STEP_DAYS}D",
    ).to_numpy()
    means = rating.position_means(logs[logs.season <= seasons[-1]])
    table = rating.skater_table(logs, means)
    fit_stints = stints[stints.season.isin(seasons)].reset_index(drop=True)
    effects, penalty = {}, {}
    for kind in onice.KINDS:
        blocks = onice.game_blocks(onice.attacking_rows(stints, kind), players)
        by_window, _ = onice.ridge(blocks, len(players), dates, PENALTIES[kind])
        errors = {}
        for p, value in enumerate(PENALTIES[kind]):
            one = {kind: {w: by_window[w][:, p] for w in onice.WINDOWS}}
            trial = rating.attach_on_ice(table, players, dates, one)
            names = rating.ON_ICE[kind]
            design = Design(fit_stints, trial, kind, names["off"] + names["def"])
            train = np.ones(len(design.season), bool)
            errors[value] = design.fit(names["off"], names["def"], "xg", train)[2]
        penalty[kind] = min(errors, key=errors.get)
        chosen = PENALTIES[kind].index(penalty[kind])
        effects[kind] = {w: by_window[w][:, chosen] for w in onice.WINDOWS}
    table = rating.attach_on_ice(table, players, dates, effects)

    weights, predicted, designs = {}, {}, {}
    names = rating.FEATURES["5v5"]
    designs["5v5"] = Design(
        fit_stints, table, "5v5", sorted(set(names["off"] + names["def"]))
    )
    train = np.ones(len(designs["5v5"].season), bool)
    predicted["5v5"], weights["5v5"], _ = designs["5v5"].fit(
        names["off"], names["def"], "xg", train
    )
    table["off5"] = rating.centered(
        rating.weighted(table, weights["5v5"]["off"]), table.toi_5v5, table
    )
    table["def5"] = rating.centered(
        rating.weighted(table, weights["5v5"]["def"], -1.0), table.toi_5v5, table
    )
    names = rating.FEATURES["pp"]
    designs["pp"] = Design(
        fit_stints, table, "pp", sorted(set(names["off"] + names["def"]))
    )
    train_pp = np.ones(len(designs["pp"].season), bool)
    predicted["pp"], weights["pp"], _ = designs["pp"].fit(
        names["off"], names["def"], "xg", train_pp
    )

    # Finishing: how hard to shrink a shooter's goals above his expected goals.
    ahead = _future(table, ["goals", "ixg", "n"])
    known = (table.season <= seasons[-1]) & (ahead.n >= MIN_FUTURE_GAMES)
    surplus = (ahead.goals - ahead.ixg)[known]
    misses = {}
    for prior in FINISHING_PRIORS:
        finishing = (table.career_goals - table.career_ixg) / (table.career_ixg + prior)
        misses[prior] = float(
            ((surplus - finishing[known] * ahead.ixg[known]) ** 2).sum()
        )
    model = {"finishing_prior_xg": min(misses, key=misses.get)}
    table["shoot"] = rating.shooting(table, model)
    shooting = {}
    for kind in onice.KINDS:
        design = Design(fit_stints, table, kind, ["shoot"])
        train = np.ones(len(design.season), bool)
        _, by_role, _ = design.fit(
            ["shoot"], [], "goals", train, predicted[kind][:, None]
        )
        shooting[kind] = {pos: by_role["off"][pos]["shoot"] for pos in ("F", "D")}
    return {
        "version": "nhl-players-v1",
        "fit_seasons": seasons,
        "penalty_hours": penalty,
        "position_means": means,
        **model,
        "weights": weights,
        "shooting": shooting,
        "goalie": goalies.fit(
            goalie_logs.load(range(HISTORY_START_SEASON, seasons[-1] + 1)), seasons[-1]
        ),
    }


def write(model: dict) -> None:
    rating.MODEL_PATH.write_text(json.dumps(model, indent=1, sort_keys=True) + "\n")
