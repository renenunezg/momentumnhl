"""Goalie rating: goals saved per hour above expected, recency weighted.

A goalie's goals saved against MoneyPuck expected goals are summed over his
earlier appearances, each one back counting less, and shrunk by a number of
league-average hours. A linear fit against his next appearances puts that on
the scale of goals saved per 60 minutes and adds an age curve: each year past
30, and each year short of 25, costs a little. Scored on every season from
2012-13 with weights fit on earlier ones, this beat the 40-appearance estimate
in backend/model/goalies.py, and the age terms kept one sign throughout.
Shot-danger recalibration, workload, rebound control and the team's other
goalies were tested and added nothing.

`reliability` is the share of a goalie's own record the rating keeps (the
rest is the league average), so a short record reads as an uncertain grade.
"""

import numpy as np
import pandas as pd

HALF_LIVES = (40.0, 80.0, 160.0, 320.0, np.inf)
PRIOR_HOURS = (20.0, 40.0, 80.0, 160.0, 320.0)
FUTURE_GAMES, MIN_FUTURE_GAMES = 25, 15
SUMS = ["n", "hours", "saved"]
PRIME_FROM, PRIME_TO = 25.0, 30.0


def _prepared(logs: pd.DataFrame, born: pd.Series, as_of=None) -> pd.DataFrame:
    """Appearances in order for each goalie, with his age terms; `born` maps
    goalie_id to a birth date, and a goalie it lacks gets no age adjustment."""
    table = logs[logs.toi > 0].copy()
    table["n"], table["hours"] = 1.0, table.toi / 3600
    table["saved"] = table.xga - table.ga
    table["now"] = False
    if as_of is not None:
        latest = (
            table.sort_values(["game_date", "game_id"]).groupby("goalie_id").tail(1)
        )
        today = latest[latest.season.eq(latest.season.max())].copy()
        today[SUMS] = 0.0
        today["game_date"], today["game_id"], today["now"] = as_of, "", True
        table = pd.concat([table, today], ignore_index=True)
    birth = pd.to_datetime(table.goalie_id.map(born))
    age = (pd.to_datetime(table.game_date) - birth).dt.days / 365.25
    table["over_30"] = (age - PRIME_TO).clip(lower=0).fillna(0.0)
    table["under_25"] = (PRIME_FROM - age).clip(lower=0).fillna(0.0)
    return table.sort_values(
        ["goalie_id", "game_date", "now", "game_id"], kind="stable"
    ).reset_index(drop=True)


def _decayed(table: pd.DataFrame, half_life: float) -> pd.DataFrame:
    """Recency-weighted sums of each goalie's earlier appearances: the latest
    counts 0.5 ** (1 / half_life), the one before that the same factor less."""
    keep = 0.5 ** (1.0 / half_life) if np.isfinite(half_life) else 1.0
    values, goalie = table[["hours", "saved"]].to_numpy(), table.goalie_id.to_numpy()
    out = np.zeros_like(values)
    for i in range(1, len(table)):
        if goalie[i] == goalie[i - 1]:
            out[i] = keep * (out[i - 1] + values[i - 1])
    return pd.DataFrame(out, columns=["hours", "saved"], index=table.index)


def fit(logs: pd.DataFrame, born: pd.Series, through: int) -> dict:
    """Half-life, prior, scale and age weights chosen on appearances in seasons
    up to `through`, against the goalie's next FUTURE_GAMES that season."""
    table = _prepared(logs, born)
    group = table.groupby(["goalie_id", "season"])
    total = group[SUMS].cumsum()
    end = total.groupby([table.goalie_id, table.season]).shift(-(FUTURE_GAMES - 1))
    whole = group[SUMS].transform("sum")
    ahead = end.where(end.n.notna(), whole) - (total - table[SUMS])
    use = ((table.season <= through) & (ahead.n >= MIN_FUTURE_GAMES)).to_numpy()
    target = (ahead.saved / ahead.hours).to_numpy()[use]
    weight = ahead.hours.to_numpy()[use]
    ages = table[["over_30", "under_25"]].to_numpy()[use]
    best = None
    for half_life in HALF_LIVES:
        sums = _decayed(table, half_life)
        for prior in PRIOR_HOURS:
            shrunk = (sums.saved / (sums.hours + prior)).to_numpy()[use]
            X = np.c_[np.ones(len(shrunk)), shrunk, ages]
            weighted = X * weight[:, None]
            beta = np.linalg.lstsq(weighted.T @ X, weighted.T @ target, rcond=None)[0]
            error = float(np.average((target - X @ beta) ** 2, weights=weight))
            if best is None or error < best[0]:
                best = (error, half_life, prior, beta)
    _, half_life, prior, beta = best
    return {
        "half_life_games": half_life,
        "prior_hours": prior,
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "per_year_over_30": float(beta[2]),
        "per_year_under_25": float(beta[3]),
    }


def rate(logs: pd.DataFrame, params: dict, born: pd.Series, as_of) -> pd.DataFrame:
    """Each goalie who has played this season: goals saved per 60 minutes
    above expected as of `as_of`, with his team and games this season."""
    table = _prepared(logs, born, as_of)
    sums = _decayed(table, params["half_life_games"])
    kept = params["slope"] / (sums.hours + params["prior_hours"])
    table["goals_saved"] = (
        params["intercept"]
        + kept * sums.saved
        + params["per_year_over_30"] * table.over_30
        + params["per_year_under_25"] * table.under_25
    )
    table["reliability"] = (kept * sums.hours).clip(0.0, 1.0)
    season = table.season.max()
    games = table[~table.now & table.season.eq(season)].groupby("goalie_id").size()
    now = table[table.now].copy()
    now["games"] = now.goalie_id.map(games).fillna(0).astype(int)
    columns = ["goalie_id", "name", "team_abbr", "games", "goals_saved", "reliability"]
    return now[columns]
