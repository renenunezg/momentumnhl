"""Who was on the ice, and what each skater did to shot quality there.

`stints` cuts every game into stretches with a fixed set of skaters and sums
each side's expected goals in them. `ridge` then regresses the attacking
side's expected goals per hour on an offense term for each attacker and a
defense term for each defender, with home ice and score state, penalized
toward zero. A skater's two numbers are therefore his effect with linemates
and opponents held constant. Power-play stints give a power-play offense term
and a penalty-kill defense term the same way.

Fits use the trailing WINDOWS days before a date, so a rating never reads a
game on or after it.
"""

import numpy as np
import pandas as pd
from scipy.linalg import solve

SLOTS = 5
HOME = [f"h{i}" for i in range(SLOTS)]
AWAY = [f"a{i}" for i in range(SLOTS)]
PERIOD_SECONDS = 1200
REGULATION_PERIODS = 3
# Skater counts kept: 5v5, and a one-man advantage either way, goalies in.
STRENGTHS = {(5, 5), (5, 4), (4, 5)}
WINDOWS = {"s": 365, "l": 730}
KINDS = ("5v5", "pp")
SCORE_STATES = 4


def _game_stints(game_id, shifts, shots, skaters, home_team) -> list[tuple]:
    period = shifts.period.to_numpy()
    start = (period - 1) * PERIOD_SECONDS + shifts.start.to_numpy()
    end = (period - 1) * PERIOD_SECONDS + shifts.end.to_numpy()
    keep = (end > start) & (period <= REGULATION_PERIODS)
    start, end = start[keep], end[keep]
    player = shifts.player_id.to_numpy()[keep]
    is_home = (shifts.team.to_numpy() == home_team)[keep]
    is_skater = np.fromiter((p in skaters for p in player), bool, len(player))
    cuts = np.unique(np.r_[start, end])
    first, last = np.searchsorted(cuts, start), np.searchsorted(cuts, end)
    count = len(cuts) - 1
    on_ice = {
        (side, role): [[] for _ in range(count)]
        for side in (True, False)
        for role in (True, False)
    }
    for p, a, b, side, role in zip(player, first, last, is_home, is_skater):
        bucket = on_ice[side, role]
        for k in range(a, b):
            bucket[k].append(p)
    # A shot at the instant of a change belongs to the stint that just ended.
    clock = shots.time.to_numpy()
    where = np.searchsorted(cuts, clock, side="left") - 1
    inside = (where >= 0) & (where < count)
    by_home = shots.home.to_numpy()
    xg, goal = shots.xg.to_numpy(), shots.goal.to_numpy()
    sums = np.zeros((count, 4))
    for column, mask, value in (
        (0, by_home, xg),
        (1, ~by_home, xg),
        (2, by_home, goal),
        (3, ~by_home, goal),
    ):
        chosen = inside & mask
        np.add.at(sums[:, column], where[chosen], value[chosen])
    scored = goal == 1
    home_goals = np.sort(clock[scored & by_home])
    away_goals = np.sort(clock[scored & ~by_home])
    home_score = np.searchsorted(home_goals, cuts[:-1], side="right")
    away_score = np.searchsorted(away_goals, cuts[:-1], side="right")
    rows = []
    for k in range(count):
        home = sorted(set(on_ice[True, True][k]))
        away = sorted(set(on_ice[False, True][k]))
        goalies = len(set(on_ice[True, False][k])), len(set(on_ice[False, False][k]))
        if goalies != (1, 1) or (len(home), len(away)) not in STRENGTHS:
            continue
        rows.append(
            (
                game_id,
                cuts[k + 1] - cuts[k],
                len(home),
                len(away),
                *(home + [0] * SLOTS)[:SLOTS],
                *(away + [0] * SLOTS)[:SLOTS],
                *sums[k],
                home_score[k],
                away_score[k],
            )
        )
    return rows


def stints(
    shifts: pd.DataFrame, shots: pd.DataFrame, logs: pd.DataFrame
) -> pd.DataFrame:
    """One row per stint for every game present in all three inputs."""
    skaters = logs.groupby("game_id").player_id.agg(set).to_dict()
    day = logs.drop_duplicates("game_id").set_index("game_id").game_date.to_dict()
    home_team = shots.drop_duplicates("game_id").set_index("game_id").home_team
    shots_of = dict(tuple(shots.groupby("game_id")))
    rows = []
    for game_id, game in shifts.groupby("game_id"):
        teams = set(game.team.unique())
        if game_id not in skaters or home_team.get(game_id) not in teams:
            continue
        if len(teams) != 2:
            raise ValueError(f"Game {game_id} does not have two teams of shifts")
        rows += _game_stints(
            game_id, game, shots_of[game_id], skaters[game_id], home_team[game_id]
        )
    columns = [
        "game_id", "secs", "nh", "na", *HOME, *AWAY,
        "xg_home", "xg_away", "g_home", "g_away", "score_home", "score_away",
    ]  # fmt: skip
    out = pd.DataFrame(rows, columns=columns)
    out["day"] = pd.to_datetime(out.game_id.map(day))
    return out


def attacking_rows(table: pd.DataFrame, kind: str) -> dict:
    """Each stint as rows with one side attacking: its ids, the defenders' ids,
    hours, expected goals and goals, and the nuisance columns (constant, home
    ice, and at 5v5 the attacker's score state)."""
    home, away = table[HOME].to_numpy(), table[AWAY].to_numpy()
    nh, na = table.nh.to_numpy(), table.na.to_numpy()
    hours = table.secs.to_numpy() / 3600
    xg_home, xg_away = table.xg_home.to_numpy(), table.xg_away.to_numpy()
    g_home, g_away = table.g_home.to_numpy(), table.g_away.to_numpy()
    if kind == "5v5":
        first = second = np.flatnonzero((nh == 5) & (na == 5))
    else:
        first = np.flatnonzero((nh == 5) & (na == 4))
        second = np.flatnonzero((nh == 4) & (na == 5))
    is_home = np.r_[np.ones(len(first)), np.zeros(len(second))]
    nuisance = [np.ones(len(is_home)), is_home]
    if kind == "5v5":
        margin = (table.score_home - table.score_away).to_numpy()
        lead = np.r_[margin[first], -margin[second]]
        nuisance += [lead <= -2, lead == -1, lead == 1, lead >= 2]
    which = np.r_[first, second]
    return {
        "off": np.r_[home[first], away[second]],
        "def": np.r_[away[first], home[second]],
        "hours": hours[which],
        "xg": np.r_[xg_home[first], xg_away[second]],
        "goals": np.r_[g_home[first], g_away[second]],
        "nuisance": np.column_stack(nuisance).astype(float),
        "game": table.game_id.to_numpy()[which],
        "day": table.day.to_numpy()[which],
        "stint": which,
    }


def game_blocks(rows: dict, players: np.ndarray) -> list[tuple]:
    """Each game's share of the normal equations, in date order: its columns,
    X'WX and X'y. Columns are offense for each of `players`, then defense for
    each, then the nuisance terms. Sums over games add, so a trailing window
    is built by adding the games that enter it and subtracting those that
    leave."""
    count = len(players)
    lookup = {p: i for i, p in enumerate(players)}
    index = np.vectorize(lambda p: lookup.get(p, -1), otypes=[np.int64])
    off, dfn = index(rows["off"]), index(rows["def"])
    dfn = np.where(dfn < 0, -1, dfn + count)
    extra = rows["nuisance"].shape[1]
    order = np.argsort(rows["game"], kind="stable")
    bounds = np.flatnonzero(np.diff(rows["game"][order])) + 1
    blocks = []
    for part in np.split(order, bounds):
        ids = np.c_[off[part], dfn[part]]
        columns, inverse = np.unique(ids[ids >= 0], return_inverse=True)
        design = np.zeros((len(part), len(columns) + extra))
        at = np.repeat(np.arange(len(part)), ids.shape[1]).reshape(ids.shape)
        design[at[ids >= 0], inverse] = 1.0
        design[:, len(columns) :] = rows["nuisance"][part]
        weight = rows["hours"][part]
        blocks.append(
            (
                rows["day"][part[0]],
                np.r_[columns, 2 * count + np.arange(extra)],
                (design * weight[:, None]).T @ design,
                design.T @ rows["xg"][part],
            )
        )
    blocks.sort(key=lambda block: block[0])
    return blocks


def ridge(blocks: list[tuple], count: int, dates, penalties) -> tuple[dict, dict]:
    """Penalized fits on the trailing windows before each of `dates` (sorted).
    Returns per window an array [date, penalty, offense or defense, player] in
    expected goals per hour, defense signed so that higher is better, and the
    hours each player was on the ice in the window."""
    extra = int((blocks[0][1] >= 2 * count).sum())
    size = 2 * count + extra
    days = np.array([block[0] for block in blocks])
    effects = {
        w: np.zeros((len(dates), len(penalties), 2, count), np.float32) for w in WINDOWS
    }
    hours = {w: np.zeros((len(dates), count), np.float32) for w in WINDOWS}
    for window, span in WINDOWS.items():
        gram, moment = np.zeros((size, size)), np.zeros(size)
        low = high = 0
        for d, date in enumerate(dates):
            new_high = np.searchsorted(days, date, side="left")
            new_low = np.searchsorted(
                days, date - np.timedelta64(span, "D"), side="left"
            )
            for sign, a, b in ((1.0, high, new_high), (-1.0, low, new_low)):
                for _, columns, block, target in blocks[a:b]:
                    gram[np.ix_(columns, columns)] += sign * block
                    moment[columns] += sign * target
            low, high = new_low, new_high
            diagonal = np.diag(gram)[: 2 * count]
            active = np.flatnonzero(diagonal > 1e-9)
            if not len(active):
                continue
            hours[window][d] = diagonal[:count]
            use = np.r_[active, 2 * count + np.arange(extra)]
            sub = gram[np.ix_(use, use)]
            penalized = np.r_[np.ones(len(active)), np.zeros(extra)]
            for p, penalty in enumerate(penalties):
                beta = solve(
                    sub + np.diag(penalty * penalized + 1e-9),
                    moment[use],
                    assume_a="pos",
                )
                full = np.zeros(2 * count)
                full[active] = beta[: len(active)]
                effects[window][d, p, 0] = full[:count]
                effects[window][d, p, 1] = -full[count:]
    return effects, hours
