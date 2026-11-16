"""Goalie rating: goals saved per hour above expected, recency weighted.

A goalie's goals saved against MoneyPuck expected goals are summed over his
earlier appearances, each one back counting less, and shrunk by a number of
league-average hours. A linear fit against his next appearances puts that on
the scale of goals saved per 60 minutes. Shot-danger recalibration, workload,
rebound control and the team's other goalies were tested and added nothing.
"""

import numpy as np
import pandas as pd

HALF_LIVES = (40.0, 80.0, 160.0, 320.0, np.inf)
PRIOR_HOURS = (20.0, 40.0, 80.0, 160.0, 320.0)
FUTURE_GAMES, MIN_FUTURE_GAMES = 25, 15
SUMS = ["n", "hours", "saved"]


def _prepared(logs: pd.DataFrame, as_of=None) -> pd.DataFrame:
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


def fit(logs: pd.DataFrame, through: int) -> dict:
    """Half-life, prior and scale chosen on appearances in seasons up to
    `through`, against the goalie's next FUTURE_GAMES appearances that season."""
    table = _prepared(logs)
    group = table.groupby(["goalie_id", "season"])
    total = group[SUMS].cumsum()
    end = total.groupby([table.goalie_id, table.season]).shift(-(FUTURE_GAMES - 1))
    ahead = end.where(end.n.notna(), group[SUMS].transform("sum")) - (
        total - table[SUMS]
    )
    use = ((table.season <= through) & (ahead.n >= MIN_FUTURE_GAMES)).to_numpy()
    target, weight = (ahead.saved / ahead.hours)[use], ahead.hours[use]
    best = None
    for half_life in HALF_LIVES:
        sums = _decayed(table, half_life)
        for prior in PRIOR_HOURS:
            rate = (sums.saved / (sums.hours + prior))[use]
            X = np.c_[np.ones(len(rate)), rate]
            beta = np.linalg.lstsq(
                (X * weight.to_numpy()[:, None]).T @ X,
                (X * weight.to_numpy()[:, None]).T @ target,
                rcond=None,
            )[0]
            error = float(np.average((target - X @ beta) ** 2, weights=weight))
            if best is None or error < best[0]:
                best = (error, half_life, prior, beta)
    _, half_life, prior, beta = best
    return {
        "half_life_games": half_life,
        "prior_hours": prior,
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
    }


def rate(logs: pd.DataFrame, params: dict, as_of) -> pd.DataFrame:
    """Each goalie who has played this season: goals saved per 60 minutes
    above expected as of `as_of`, with his team and games this season."""
    table = _prepared(logs, as_of)
    sums = _decayed(table, params["half_life_games"])
    shrunk = sums.saved / (sums.hours + params["prior_hours"])
    table["goals_saved"] = params["intercept"] + params["slope"] * shrunk
    season = table.season.max()
    games = table[~table.now & table.season.eq(season)].groupby("goalie_id").size()
    now = table[table.now].copy()
    now["games"] = now.goalie_id.map(games).fillna(0).astype(int)
    return now[["goalie_id", "name", "team_abbr", "games", "goals_saved"]]
