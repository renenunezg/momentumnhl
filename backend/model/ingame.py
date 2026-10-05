"""In-game home win probability from score, clock, manpower, and the pregame
goal rates.

The score margin is a Markov chain. Each side scores at its pregame rate
times a fitted league multiplier for the score state and the time left, so a
tied game slowing down, a trailing team pulling its goalie, and the empty-net
goals that follow are all in the transition rates. Backward induction from
the horn gives the value of every margin at every moment; a regulation tie is
worth the same even overtime share the pregame model uses. The current power
play or empty net is then played out on top of that table with its own fitted
multipliers. Every function is vectorized; the live publisher passes arrays
of length one.

The published multipliers live in backend/data_static/ingame_model.json: the
backtest's holdout fit, copied there by hand so a rerun cannot move the live
model unreviewed.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import poisson

from backend.config import STATIC_DIR

VERSION = "nhl-ingame-v1"
PARAMETERS_PATH = STATIC_DIR / "ingame_model.json"
REGULATION_SECONDS = 3600
OVERTIME_SECONDS = 300
OVERTIME_SHARE = 0.5
STEP_SECONDS = 5
STEPS = REGULATION_SECONDS // STEP_SECONDS
MAX_MARGIN = 10
# A manpower state still on the ice after its penalty should have expired.
MIN_ADVANTAGE_SECONDS = 10.0
# Pseudo-goals at the pregame rate that steady a thin score-state bucket.
RATE_PRIOR_GOALS = 5.0

FULL, HOME_ADV1, HOME_ADV2, AWAY_ADV1, AWAY_ADV2, EVEN_SHORT = range(6)
HOME_NET_EMPTY, AWAY_NET_EMPTY = 6, 7
MANPOWER_KINDS = (HOME_ADV1, HOME_ADV2, AWAY_ADV1, AWAY_ADV2, EVEN_SHORT)
NET_EMPTY_KINDS = (HOME_NET_EMPTY, AWAY_NET_EMPTY)

# Seconds of regulation remaining that bound the score-state rate buckets.
RATE_EDGES = (0, 30, 60, 90, 120, 180, 240, 300, 600, 1200, 2400, 3600)
# Tied, trailing by 1, 2, 3 or more, leading by 1, 2, 3 or more.
SIDE_STATES = 7
_MARGINS = np.arange(-MAX_MARGIN, MAX_MARGIN + 1)


def kinds(situation_code, seconds_elapsed, margin, overtime) -> np.ndarray:
    """Manpower kind from the NHL situation code (away goalie, away skaters,
    home skaters, home goalie). A net counts as empty only for a trailing
    team in the third period; a goalie pulled on a delayed penalty is brief
    and treated as full strength."""
    code = pd.Series(situation_code, dtype="string")
    if not code.str.fullmatch(r"[01][0-9][0-9][01]").all():
        raise ValueError("Unknown situation code")
    digits = code.str.extract(r"(.)(.)(.)(.)").astype(int).to_numpy()
    away_goalie, away_skaters, home_skaters, home_goalie = digits.T
    elapsed = np.asarray(seconds_elapsed, dtype=float)
    margin = np.asarray(margin)
    overtime = np.asarray(overtime, dtype=bool)
    difference = home_skaters - away_skaters
    out = np.full(len(code), FULL)
    goalies_in = (away_goalie == 1) & (home_goalie == 1)
    out[goalies_in & (difference == 1)] = HOME_ADV1
    out[goalies_in & (difference >= 2)] = HOME_ADV2
    out[goalies_in & (difference == -1)] = AWAY_ADV1
    out[goalies_in & (difference <= -2)] = AWAY_ADV2
    out[goalies_in & (difference == 0) & (home_skaters < 5) & ~overtime] = EVEN_SHORT
    # Overtime carries one fitted advantage, whatever the skater counts.
    out[goalies_in & overtime & (difference > 0)] = HOME_ADV1
    out[goalies_in & overtime & (difference < 0)] = AWAY_ADV1
    late = ~overtime & (elapsed >= 2400)
    out[late & (home_goalie == 0) & (away_goalie == 1) & (margin < 0)] = HOME_NET_EMPTY
    out[late & (away_goalie == 0) & (home_goalie == 1) & (margin > 0)] = AWAY_NET_EMPTY
    return out


def _side_state(margin) -> np.ndarray:
    margin = np.asarray(margin)
    state = np.zeros(margin.shape, dtype=int)
    state[margin < 0] = np.minimum(-margin[margin < 0], 3)
    state[margin > 0] = 3 + np.minimum(margin[margin > 0], 3)
    return state


def _values(rates: np.ndarray, home_lambda, away_lambda) -> np.ndarray:
    """Home win probability for each game, steps remaining, and margin."""
    values = np.empty((len(home_lambda), STEPS + 1, len(_MARGINS)), dtype=np.float32)
    values[:, 0] = np.where(
        _MARGINS > 0, 1.0, np.where(_MARGINS == 0, OVERTIME_SHARE, 0)
    )
    home_state, away_state = _side_state(_MARGINS), _side_state(-_MARGINS)
    home_step = home_lambda[:, None] / REGULATION_SECONDS * STEP_SECONDS
    away_step = away_lambda[:, None] / REGULATION_SECONDS * STEP_SECONDS
    for step in range(1, STEPS + 1):
        bucket = np.searchsorted(RATE_EDGES, (step - 0.5) * STEP_SECONDS, "right") - 1
        home = home_step * rates[home_state, bucket]
        away = away_step * rates[away_state, bucket]
        later = values[:, step - 1]
        # The widest margins are decided; a further goal changes nothing.
        up = np.concatenate([later[:, 1:], later[:, -1:]], axis=1)
        down = np.concatenate([later[:, :1], later[:, :-1]], axis=1)
        values[:, step] = home * up + away * down + (1 - home - away) * later
    return values


def _lookup(values, game, margin, seconds) -> np.ndarray:
    position = np.clip(seconds, 0, REGULATION_SECONDS) / STEP_SECONDS
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, STEPS)
    column = np.clip(margin, -MAX_MARGIN, MAX_MARGIN) + MAX_MARGIN
    weight = position - low
    return (1 - weight) * values[game, low, column] + weight * values[
        game, high, column
    ]


def home_win_probability(
    params: dict, *, game, margin, seconds_remaining, overtime, kind,
    advantage_seconds, home_lambda, away_lambda,
) -> np.ndarray:  # fmt: skip
    """`game` groups states that share one pregame anchor."""
    margin = np.asarray(margin, dtype=int)
    overtime = np.asarray(overtime, dtype=bool)
    kind = np.asarray(kind, dtype=int)
    remaining = np.where(overtime, 0.0, np.asarray(seconds_remaining, dtype=float))
    advantage = np.asarray(advantage_seconds, dtype=float)
    home_lambda = np.asarray(home_lambda, dtype=float)
    away_lambda = np.asarray(away_lambda, dtype=float)
    if (
        (remaining < 0).any() or (remaining > REGULATION_SECONDS).any()
        or not np.isfinite(home_lambda + away_lambda + remaining).all()
        or (home_lambda <= 0).any() or (away_lambda <= 0).any()
        or (overtime & (margin != 0)).any()
    ):  # fmt: skip
        raise ValueError("Invalid in-game state")
    _, first, index = np.unique(
        np.asarray(game), return_index=True, return_inverse=True
    )
    values = _values(
        np.asarray(params["rates"], dtype=float), home_lambda[first], away_lambda[first]
    )
    home_rate = home_lambda / REGULATION_SECONDS
    away_rate = away_lambda / REGULATION_SECONDS
    out = _lookup(values, index, margin, remaining)
    regulation = ~overtime

    # A power play: its goals, then the table from where it leaves the game.
    window = np.minimum(np.maximum(advantage, MIN_ADVANTAGE_SECONDS), remaining)
    even = {"for": params["even_short"], "against": params["even_short"]}
    for selected, sign, factors in (
        (HOME_ADV1, 1, params["advantage_1"]),
        (AWAY_ADV1, -1, params["advantage_1"]),
        (HOME_ADV2, 1, params["advantage_2"]),
        (AWAY_ADV2, -1, params["advantage_2"]),
        (EVEN_SHORT, 1, even),
    ):
        mask = regulation & (kind == selected)
        if mask.any():
            ahead, behind = (
                (home_rate, away_rate) if sign == 1 else (away_rate, home_rate)
            )
            scoring = factors["for"] * ahead[mask]
            # A goal by the advantaged side ends a minor early.
            length = (
                window[mask]
                if selected == EVEN_SHORT
                else -np.expm1(-scoring * window[mask]) / scoring
            )
            scored = scoring * length
            conceded = factors["against"] * behind[mask] * length
            total = np.zeros(mask.sum())
            mass = np.zeros(mask.sum())
            for goals_for in range(4):
                for goals_against in range(3):
                    weight = poisson.pmf(goals_for, scored) * poisson.pmf(
                        goals_against, conceded
                    )
                    total += weight * _lookup(
                        values, index[mask],
                        margin[mask] + sign * (goals_for - goals_against),
                        remaining[mask] - length,
                    )  # fmt: skip
                    mass += weight
            out[mask] = total / mass

    # An empty net lasts until the next goal, which returns play to the table.
    for selected, sign in ((HOME_NET_EMPTY, 1), (AWAY_NET_EMPTY, -1)):
        rows = np.flatnonzero(regulation & (kind == selected))
        if not len(rows):
            continue
        attacker, defender = (
            (home_rate, away_rate) if sign == 1 else (away_rate, home_rate)
        )
        scoring = params["net_empty"]["attacking"] * attacker[rows] * STEP_SECONDS
        conceding = params["net_empty"]["defending"] * defender[rows] * STEP_SECONDS
        steps = np.ceil(remaining[rows] / STEP_SECONDS).astype(int)
        value = _lookup(values, index[rows], margin[rows], np.zeros(len(rows)))
        for step in range(1, steps.max() + 1):
            left = np.full(len(rows), (step - 1) * STEP_SECONDS)
            played = (
                scoring * _lookup(values, index[rows], margin[rows] + sign, left)
                + conceding * _lookup(values, index[rows], margin[rows] - sign, left)
                + (1 - scoring - conceding) * value
            )
            value = np.where(steps >= step, played, value)
        out[rows] = value

    # Sudden death: the advantaged side wins if it scores first inside the
    # advantage; otherwise the game is even again.
    out[overtime] = OVERTIME_SHARE
    span = np.clip(advantage, MIN_ADVANTAGE_SECONDS, OVERTIME_SECONDS)
    for selected, sign in ((HOME_ADV1, 1), (AWAY_ADV1, -1)):
        mask = overtime & (kind == selected)
        ahead, behind = (home_rate, away_rate) if sign == 1 else (away_rate, home_rate)
        scoring = params["overtime"]["for"] * ahead[mask]
        conceding = params["overtime"]["against"] * behind[mask]
        decided = -np.expm1(-(scoring + conceding) * span[mask])
        first_goal = scoring / (scoring + conceding) * decided + OVERTIME_SHARE * (
            1 - decided
        )
        out[mask] = first_goal if sign == 1 else 1 - first_goal
    return out


def load() -> dict:
    params = json.loads(PARAMETERS_PATH.read_text())
    if params["version"] != VERSION:
        raise ValueError(f"In-game parameters are not {VERSION}")
    return params


def neutral() -> dict:
    """Score and time only: pregame rates throughout, manpower ignored."""
    return {
        "version": VERSION,
        "rates": np.ones((SIDE_STATES, len(RATE_EDGES) - 1)).tolist(),
        "advantage_1": {"for": 1.0, "against": 1.0},
        "advantage_2": {"for": 1.0, "against": 1.0},
        "even_short": 1.0,
        "net_empty": {"attacking": 1.0, "defending": 1.0},
        "overtime": {"for": 1.0, "against": 1.0},
    }


def fit(
    intervals: pd.DataFrame, *, score_states: bool = True, manpower: bool = True
) -> dict:
    """League multipliers as goals over rate-weighted exposure. `intervals`
    holds one row per stretch between events: seconds, kind, overtime, margin
    and seconds_remaining at its end, both pregame lambdas, and whether it
    ended in a home or away goal."""
    home_exposure = intervals.seconds * intervals.home_lambda / REGULATION_SECONDS
    away_exposure = intervals.seconds * intervals.away_lambda / REGULATION_SECONDS
    regulation = ~intervals.overtime
    params = neutral()

    def ratio(goals: float, exposure: float) -> float:
        if goals <= 0 or exposure <= 0:
            raise ValueError("A manpower state has no training goals")
        return float(goals / exposure)

    def sides(home_kind: int, away_kind: int, scope) -> dict:
        home = scope & (intervals.kind == home_kind)
        away = scope & (intervals.kind == away_kind)
        return {
            "for": ratio(
                intervals.home_goal[home].sum() + intervals.away_goal[away].sum(),
                home_exposure[home].sum() + away_exposure[away].sum(),
            ),
            "against": ratio(
                intervals.away_goal[home].sum() + intervals.home_goal[away].sum(),
                away_exposure[home].sum() + home_exposure[away].sum(),
            ),
        }

    if manpower:
        params["advantage_1"] = sides(HOME_ADV1, AWAY_ADV1, regulation)
        params["advantage_2"] = sides(HOME_ADV2, AWAY_ADV2, regulation)
        even = regulation & (intervals.kind == EVEN_SHORT)
        params["even_short"] = ratio(
            (intervals.home_goal + intervals.away_goal)[even].sum(),
            (home_exposure + away_exposure)[even].sum(),
        )
        empty = sides(HOME_NET_EMPTY, AWAY_NET_EMPTY, regulation)
        params["net_empty"] = {"attacking": empty["for"], "defending": empty["against"]}
        params["overtime"] = sides(HOME_ADV1, AWAY_ADV1, intervals.overtime)
    if score_states:
        # Every manpower state, empty nets included, so the table carries
        # the goalie pull that is still coming.
        bucket = np.searchsorted(RATE_EDGES, intervals.seconds_remaining, "right") - 1
        bucket = np.clip(bucket, 0, len(RATE_EDGES) - 2)
        margin = intervals.margin.to_numpy()
        goals = np.zeros((SIDE_STATES, len(RATE_EDGES) - 1))
        exposure = np.zeros_like(goals)
        scope = regulation.to_numpy()
        for state, scored, exposed in (
            (_side_state(margin), intervals.home_goal, home_exposure),
            (_side_state(-margin), intervals.away_goal, away_exposure),
        ):
            np.add.at(goals, (state[scope], bucket[scope]), scored.to_numpy()[scope])
            np.add.at(
                exposure, (state[scope], bucket[scope]), exposed.to_numpy()[scope]
            )
        params["rates"] = (
            (goals + RATE_PRIOR_GOALS) / (exposure + RATE_PRIOR_GOALS)
        ).tolist()
    params["training"] = {
        "games": int(intervals.game_id.nunique()),
        "seasons": sorted(int(s) for s in intervals.season.unique()),
    }
    return params
