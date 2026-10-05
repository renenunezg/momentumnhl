"""Chronological backtest of the in-game win probability. Never writes a
database.

Pregame goal rates are the walk-forward backtest's, so no state sees a later
game. Multipliers are fitted on 2022 and 2023, manpower handling is kept only
if it lowers 2024 log loss, and 2025 is scored with that choice frozen. Every
game carries a total weight of one.
"""

import json

import numpy as np
import pandas as pd

from backend.config import PROCESSED_DIR
from backend.etl import play_by_play, store
from backend.model import ingame

FIT_SEASONS = (2022, 2023)
DEVELOPMENT_SEASON = 2024
HOLDOUT_SEASON = 2025
OUTPUT_DIR = PROCESSED_DIR / "ingame"
MANPOWER_PENALTIES = ("MIN", "BEN", "MAJ")


def timelines(events: pd.DataFrame, anchors: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-event score, clock, manpower code, and penalty clocks for every
    game whose play-by-play reconciles with the official final."""
    frame = events[events.period_type.isin(["REG", "OT"])].copy()
    frame = frame.sort_values(["game_id", "event_index"], kind="stable")
    frame["overtime"] = frame.period_type.eq("OT")
    frame["elapsed"] = (frame.period - 1) * 1200 + frame.period_seconds
    game = frame.groupby("game_id", sort=False)
    goal = frame.event_type.eq("goal")
    for side in ("home", "away"):
        after = frame[f"{side}_score"].where(goal)
        frame[f"{side}_after"] = (
            after.groupby(frame.game_id, sort=False).ffill().fillna(0).astype(int)
        )
        frame[f"{side}_before"] = (
            frame.groupby("game_id", sort=False)[f"{side}_after"]
            .shift(1)
            .fillna(0)
            .astype(int)
        )
        scored = goal & frame.owner_is_home.eq(side == "home")
        frame[f"{side}_goal"] = scored.fillna(False).astype(int)
        # The latest manpower penalty on this side dates its current
        # disadvantage; misconducts and penalty shots remove no skater.
        penalty = (
            frame.event_type.eq("penalty")
            & frame.owner_is_home.eq(side == "home").fillna(False)
            & frame.penalty_type.isin(MANPOWER_PENALTIES)
        )
        expiry = (frame.elapsed + frame.penalty_minutes.astype(float) * 60).where(
            penalty
        )
        frame[f"{side}_expiry"] = expiry.groupby(frame.game_id, sort=False).ffill()
    # A code describes play up to its event, so an event without one takes
    # the next coded event's.
    period = [frame.game_id, frame.period]
    frame["code"] = (
        frame.situation_code.groupby(period, sort=False).bfill().fillna("1551")
    )
    frame["next_code"] = game.code.shift(-1)
    frame["next_overtime"] = game.overtime.shift(-1)
    frame["seconds"] = (frame.elapsed - game.elapsed.shift(1).fillna(0)).clip(lower=0)

    last = frame.groupby("game_id", sort=False).tail(1).set_index("game_id")
    official = anchors.set_index("game_id")
    last = last.join(official[["home_goals", "away_goals", "last_period_type"]])
    shootout = last.last_period_type.eq("SO")
    home_won = last.home_goals > last.away_goals
    # The official final adds one shootout goal to the winner.
    valid = (
        last.home_after.eq(last.home_goals - (shootout & home_won))
        & last.away_after.eq(last.away_goals - (shootout & ~home_won))
        & last.overtime.eq(last.last_period_type.ne("REG"))
    )
    moved = (frame.home_goal + frame.away_goal).ne(
        (frame.home_after - frame.home_before) + (frame.away_after - frame.away_before)
    )
    valid &= ~moved.groupby(frame.game_id, sort=False).any().reindex(valid.index)
    frame = frame[frame.game_id.isin(valid.index[valid])]
    frame = frame.merge(
        anchors[["game_id", "home_lambda", "away_lambda", "home_win_prob"]].assign(
            home_win=(anchors.home_goals > anchors.away_goals).astype(int)
        ),
        on="game_id",
        validate="many_to_one",
    )
    quality = {
        "source_games": int(events.game_id.nunique()),
        "eligible_games": int(frame.game_id.nunique()),
        "rejected_games": int((~valid).sum()),
        "rejection": "Play-by-play disagrees with the official final",
    }
    return frame, quality


def intervals(frame: pd.DataFrame) -> pd.DataFrame:
    """The stretch of play that ends at each event, for fitting multipliers."""
    out = pd.DataFrame(
        {
            "game_id": frame.game_id,
            "season": frame.season,
            "seconds": frame.seconds,
            "overtime": frame.overtime,
            "margin": frame.home_before - frame.away_before,
            "seconds_remaining": (ingame.REGULATION_SECONDS - frame.elapsed).clip(
                lower=0
            ),
            "home_lambda": frame.home_lambda,
            "away_lambda": frame.away_lambda,
            "home_goal": frame.home_goal,
            "away_goal": frame.away_goal,
        }
    )
    out["kind"] = ingame.kinds(frame.code, frame.elapsed, out.margin, frame.overtime)
    return out[out.seconds > 0].reset_index(drop=True)


def _advantage(states: pd.DataFrame) -> np.ndarray:
    """Seconds left on the disadvantaged side's latest manpower penalty."""
    home = (states.home_expiry - states.elapsed).fillna(0).to_numpy()
    away = (states.away_expiry - states.elapsed).fillna(0).to_numpy()
    kind = states.kind.to_numpy()
    home_short = np.isin(kind, (ingame.AWAY_ADV1, ingame.AWAY_ADV2))
    away_short = np.isin(kind, (ingame.HOME_ADV1, ingame.HOME_ADV2))
    return np.where(
        home_short, home, np.where(away_short, away, np.maximum(home, away))
    )


def states(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluation states: every minute of regulation, the moment after each
    goal and penalty, the moment a manpower code first appears, and the start
    of overtime."""
    base = ["game_id", "season", "home_lambda", "away_lambda", "home_win_prob",
            "home_win"]  # fmt: skip
    game = frame.groupby("game_id", sort=False)

    def snapshot(rows, *, after: bool, code: str, overtime: str) -> pd.DataFrame:
        suffix = "after" if after else "before"
        out = rows[base + ["elapsed", "home_expiry", "away_expiry"]].copy()
        out["home"], out["away"] = rows[f"home_{suffix}"], rows[f"away_{suffix}"]
        out["code"], out["overtime"] = rows[code], rows[overtime]
        return out

    played = frame.event_type.isin(["goal", "penalty"]) & frame.next_code.notna()
    # A sudden-death goal ends the game; nothing is left to forecast.
    played &= ~(frame.overtime & frame.event_type.eq("goal"))
    following = snapshot(
        frame[played], after=True, code="next_code", overtime="next_overtime"
    )
    changed = frame.code.ne(game.code.shift(1)) & game.cumcount().gt(0)
    arriving = snapshot(frame[changed], after=False, code="code", overtime="overtime")
    # The penalty that caused a new code is the latest one before the event.
    for side in ("home", "away"):
        arriving[f"{side}_expiry"] = game[f"{side}_expiry"].shift(1)[changed]

    minutes = pd.DataFrame(
        {"elapsed": np.arange(0, ingame.REGULATION_SECONDS, 60)}
    ).merge(frame[base].drop_duplicates("game_id"), how="cross")
    regulation = frame[~frame.overtime].sort_values("elapsed", kind="stable")
    minutes = minutes.sort_values("elapsed", kind="stable")
    past = pd.merge_asof(
        minutes,
        regulation[["game_id", "elapsed", "home_after", "away_after", "home_expiry",
                    "away_expiry"]],
        on="elapsed", by="game_id",
    )  # fmt: skip
    ahead = pd.merge_asof(
        minutes[["game_id", "elapsed"]],
        regulation[["game_id", "elapsed", "code"]],
        on="elapsed", by="game_id", direction="forward", allow_exact_matches=False,
    )  # fmt: skip
    past["home"] = past.home_after.fillna(0).astype(int)
    past["away"] = past.away_after.fillna(0).astype(int)
    past["code"] = ahead.code.fillna("1551").to_numpy()
    past["overtime"] = False
    past = past.drop(columns=["home_after", "away_after"])

    extra = frame[frame.overtime].groupby("game_id", sort=False).head(1)
    extra = snapshot(extra, after=False, code="code", overtime="overtime")
    extra["elapsed"] = ingame.REGULATION_SECONDS
    extra["code"] = "1331"
    extra["home_expiry"] = extra["away_expiry"] = np.nan

    out = pd.concat([past, following, arriving, extra], ignore_index=True)
    out["overtime"] = out.overtime.astype(bool)
    # A regulation event at the horn of a tied game belongs to overtime.
    out = out[out.overtime | (out.elapsed < ingame.REGULATION_SECONDS)]
    out["margin"] = out.home - out.away
    out = out[~(out.overtime & out.margin.ne(0))]
    out["seconds_remaining"] = (ingame.REGULATION_SECONDS - out.elapsed).clip(lower=0)
    out["kind"] = ingame.kinds(out.code, out.elapsed, out.margin, out.overtime)
    out["advantage_seconds"] = _advantage(out)
    out = out.drop_duplicates(
        ["game_id", "elapsed", "overtime", "home", "away", "code", "advantage_seconds"]
    )
    return out.sort_values(["game_id", "elapsed"], kind="stable").reset_index(drop=True)


def predict(params: dict, frame: pd.DataFrame) -> np.ndarray:
    return ingame.home_win_probability(
        params,
        game=frame.game_id.to_numpy(),
        margin=frame.margin.to_numpy(),
        seconds_remaining=frame.seconds_remaining.to_numpy(),
        overtime=frame.overtime.to_numpy(),
        kind=frame.kind.to_numpy(),
        advantage_seconds=frame.advantage_seconds.to_numpy(),
        home_lambda=frame.home_lambda.to_numpy(),
        away_lambda=frame.away_lambda.to_numpy(),
    )


def game_losses(frame: pd.DataFrame, probability) -> pd.DataFrame:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    y = frame.home_win.to_numpy()
    losses = pd.DataFrame(
        {
            "game_id": frame.game_id.to_numpy(),
            "log_loss": -(y * np.log(p) + (1 - y) * np.log1p(-p)),
            "brier": (p - y) ** 2,
        }
    )
    return losses.groupby("game_id").mean()


def metrics(frame: pd.DataFrame, probability) -> dict:
    losses = game_losses(frame, probability)
    weight = 1 / frame.groupby("game_id").game_id.transform("size").to_numpy()
    p = np.asarray(probability, dtype=float)
    y = frame.home_win.to_numpy()
    bucket = np.minimum((p * 10).astype(int), 9)
    calibration = [
        {
            "bucket": int(b),
            "states": int((bucket == b).sum()),
            "games": int(frame.game_id[bucket == b].nunique()),
            "predicted": float(np.average(p[bucket == b], weights=weight[bucket == b])),
            "observed": float(np.average(y[bucket == b], weights=weight[bucket == b])),
        }
        for b in range(10)
        if (bucket == b).any()
    ]
    return {
        "games": int(len(losses)),
        "states": int(len(frame)),
        "log_loss": float(losses.log_loss.mean()),
        "brier": float(losses.brier.mean()),
        "calibration": calibration,
    }


def paired(frame: pd.DataFrame, baseline, candidate) -> dict:
    """Candidate minus baseline, with a game bootstrap interval."""
    delta = game_losses(frame, candidate) - game_losses(frame, baseline)
    rng = np.random.default_rng(20261001)
    values = delta.to_numpy()
    draws = np.array(
        [
            values[rng.integers(0, len(values), len(values))].mean(axis=0)
            for _ in range(2000)
        ]
    )
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return {
        name: {
            "delta": float(delta[name].mean()),
            "ci95": [float(low[i]), float(high[i])],
        }
        for i, name in enumerate(delta.columns)
    }


def _models(fit_intervals: pd.DataFrame) -> dict:
    return {
        "score_time": ingame.neutral(),
        "score_states": ingame.fit(fit_intervals, manpower=False),
        "score_states_manpower": ingame.fit(fit_intervals),
    }


def _summary(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "calibration"}


def run() -> dict:
    anchors = store.read_processed("backtest.parquet")
    anchors["game_id"] = anchors.game_id.astype(str)
    events = play_by_play.read(sorted(anchors.season.unique()))
    frame, quality = timelines(events, anchors)
    stretches, evaluated = intervals(frame), states(frame)

    development = evaluated[evaluated.season.eq(DEVELOPMENT_SEASON)]
    models = _models(stretches[stretches.season.isin(FIT_SEASONS)])
    selection = {
        name: _summary(metrics(development, predict(params, development)))
        for name, params in models.items()
    }
    chosen = min(
        ("score_states", "score_states_manpower"),
        key=lambda n: selection[n]["log_loss"],
    )

    # The choice is frozen; 2025 was used for neither fitting nor selection.
    training = stretches[stretches.season < HOLDOUT_SEASON]
    final = _models(training)
    holdout = evaluated[evaluated.season.eq(HOLDOUT_SEASON)].reset_index(drop=True)
    if set(training.game_id) & set(holdout.game_id):
        raise ValueError("Training and holdout games overlap")
    predictions = {
        "pregame": holdout.home_win_prob.to_numpy(),
        "score_time": predict(final["score_time"], holdout),
        "chosen": predict(final[chosen], holdout),
    }
    third = holdout.elapsed >= 2400
    slices = {
        "all": np.ones(len(holdout), dtype=bool),
        "period_1": (holdout.elapsed < 1200).to_numpy(),
        "period_2": holdout.elapsed.between(1200, 2399).to_numpy(),
        "period_3": (third & ~holdout.overtime).to_numpy(),
        "late_close": (
            third & ~holdout.overtime & holdout.margin.abs().le(1)
        ).to_numpy(),
        "manpower": holdout.kind.isin(ingame.MANPOWER_KINDS).to_numpy(),
        "net_empty": holdout.kind.isin(ingame.NET_EMPTY_KINDS).to_numpy(),
        "overtime": holdout.overtime.to_numpy(),
    }
    report_slices = {}
    for name, mask in slices.items():
        part = holdout[mask]
        report_slices[name] = {
            **{k: metrics(part, p[mask]) for k, p in predictions.items()},
            "chosen_vs_score_time": paired(
                part, predictions["score_time"][mask], predictions["chosen"][mask]
            ),
        }
    opening = holdout[holdout.elapsed.eq(0) & ~holdout.overtime]
    start = predict(final[chosen], opening)
    report = {
        "data": quality,
        "fit_seasons": list(FIT_SEASONS),
        "development_season": DEVELOPMENT_SEASON,
        "holdout_season": HOLDOUT_SEASON,
        "selection": selection,
        "chosen": chosen,
        "parameters": final[chosen],
        # The table reshapes scoring, so puck drop is near, not equal to,
        # the published pregame probability.
        "puck_drop_difference": {
            "mean": float(np.abs(start - opening.home_win_prob).mean()),
            "max": float(np.abs(start - opening.home_win_prob).max()),
        },
        "holdout": report_slices,
        "limitations": [
            "Advantage time is dated from the latest penalty; stacked ones are rough",
            "Replay uses corrected final records, not measured live feed latency",
            "No live market comparison",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "backtest.json").write_text(json.dumps(report, indent=1) + "\n")
    (OUTPUT_DIR / "parameters.json").write_text(
        json.dumps(final[chosen], indent=1) + "\n"
    )
    store.write_processed(
        holdout.assign(**{f"p_{k}": v for k, v in predictions.items()}),
        "ingame",
        "holdout.parquet",
    )
    return report
