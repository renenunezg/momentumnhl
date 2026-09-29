"""Chronological model selection and retrospective holdout evaluation.

2022 produces initial out-of-season predictions; 2023-24 select a candidate;
2025 evaluates the locked choice. Never selects on 2025 or 2026 outcomes.
The historical MoneyPuck release can contain retrospective xG revisions.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backend import backtest
from backend.etl import goalies as goalie_etl
from backend.etl import moneypuck, store
from backend.model import calibration, goalies, signals
from backend.model.signals import Specification

TRAIN_SEASON = 2022
DEVELOPMENT_SEASONS = (2023, 2024)
HOLDOUT_SEASON = 2025


def score(frame):
    return {k: v for k, v in backtest.metrics(frame).items() if k != "calibration"}


def calibrate(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    out, fitted = [], []
    for season, current in frame.groupby("season", sort=True):
        past = frame[frame.season < season]
        probability_fit = calibration.fit_probability(
            past.home_win_prob, (past.home_goals > past.away_goals).astype(int)
        )
        total_fit = calibration.fit_total(
            past.model_total, past.home_goals + past.away_goals
        )
        current = current.copy()
        current["home_win_prob"] = calibration.probability(
            current.home_win_prob, probability_fit
        )
        current["model_total"] += total_fit["offset"]
        current["calibration_train_through"] = (
            int(past.season.max()) if len(past) else None
        )
        fitted.append(
            {"season": int(season), "probability": probability_fit, "total": total_fit}
        )
        out.append(current)
    return pd.concat(out, ignore_index=True), fitted


def compare(frame, baseline, iterations=2000):
    paired = frame.merge(
        baseline, on="game_id", suffixes=("", "_base"), validate="one_to_one"
    )
    if len(paired) != len(frame):
        raise ValueError("Baseline coverage does not match candidate coverage")
    y = (paired.home_goals > paired.away_goals).astype(float).to_numpy()
    p = paired.home_win_prob.to_numpy().clip(1e-6, 1 - 1e-6)
    q = paired.home_win_prob_base.to_numpy().clip(1e-6, 1 - 1e-6)
    actual = (paired.home_goals + paired.away_goals).to_numpy()
    paired["log_loss_delta"] = -(y * np.log(p) + (1 - y) * np.log(1 - p)) + (
        y * np.log(q) + (1 - y) * np.log(1 - q)
    )
    paired["total_mae_delta"] = np.abs(paired.model_total - actual) - np.abs(
        paired.model_total_base - actual
    )
    blocks = paired.groupby("game_date")
    sizes = blocks.size().to_numpy()
    rng = np.random.default_rng(20260929)
    idx = rng.integers(0, len(sizes), (iterations, len(sizes)))
    result = {}
    for metric in ("log_loss_delta", "total_mae_delta"):
        totals = blocks[metric].sum().to_numpy()
        draws = totals[idx].sum(axis=1) / sizes[idx].sum(axis=1)
        result[metric] = {
            "mean": float(paired[metric].mean()),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    return result


def constant_baseline(fixtures):
    rows = []
    for season, current in fixtures.groupby("season", sort=True):
        past = fixtures[fixtures.season < season]
        if past.empty:
            continue
        current = current.copy()
        current["home_win_prob"] = (past.home_goals > past.away_goals).mean()
        current["model_total"] = (past.home_goals + past.away_goals).mean()
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def run(output: Path, *, include_goalies: bool = True) -> dict:
    games = moneypuck.read_games(range(2021, HOLDOUT_SEASON + 1))
    fixtures = pd.concat(
        [
            backtest.official_results(games, s)
            for s in range(TRAIN_SEASON, HOLDOUT_SEASON + 1)
        ],
        ignore_index=True,
    )
    fixtures["forecast_at"] = pd.to_datetime(fixtures.game_date, utc=True)
    features = signals.prepare(games, fixtures)
    if include_goalies:
        history = goalie_etl.load(range(2021, HOLDOUT_SEASON + 1))
        features = features.merge(
            goalies.forecast(history, fixtures), on="game_id", validate="one_to_one"
        )
    specs = [
        Specification(xg, prior, venue)
        for xg in (0.0, 0.5, 1.0)
        for prior in (0, 10, 25)
        for venue in (0.0, 0.5, 1.0)
    ]
    # Predeclared ablations, added before any holdout scoring.
    specs += [replace(s, overtime=True) for s in specs]
    if include_goalies:
        specs += [
            replace(s, goalie_weight=w)
            for s in list(specs)
            if s.xg_weight == 1 and s.prior_games == 25 and s.venue_weight == 0.5
            for w in (0.5, 1.0)
        ]
    predictions, fit_records, development = {}, {}, []
    for spec in specs:
        raw = signals.predict(features, spec)
        adjusted, fits = calibrate(raw)
        for suffix, frame in (("", raw), ("-calibrated", adjusted)):
            name = spec.name + suffix
            predictions[name] = frame
            fit_records[name] = fits if suffix else []
            dev = frame[frame.season.isin(DEVELOPMENT_SEASONS)]
            metrics = score(dev)
            development.append(
                {
                    "name": name,
                    "specification": asdict(spec),
                    "calibrated": bool(suffix),
                    **metrics,
                }
            )
    # Select separately for probability quality and total MAE. No holdout metric
    # is calculated until both choices and their settings have been recorded.
    probability_choice = min(
        development, key=lambda row: (row["log_loss"], row["name"])
    )
    total_choice = min(development, key=lambda row: (row["total_mae"], row["name"]))
    locked = {
        "probability": probability_choice,
        "total": total_choice,
        "development_seasons": list(DEVELOPMENT_SEASONS),
        "holdout_season": HOLDOUT_SEASON,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(json.dumps(locked, indent=2) + "\n")
    original = backtest.run(games)
    original_holdout = original[original.season.eq(HOLDOUT_SEASON)]
    baseline = constant_baseline(fixtures)
    baseline = baseline[baseline.season.eq(HOLDOUT_SEASON)]
    holdouts = []
    for row in development:
        name = row["name"]
        held = predictions[name].query("season == @HOLDOUT_SEASON")
        holdouts.append({"name": name, **score(held)})
    selected = {}
    for target, choice in (
        ("probability", probability_choice),
        ("total", total_choice),
    ):
        frame = predictions[choice["name"]]
        held = frame[frame.season.eq(HOLDOUT_SEASON)]
        selected[target] = {
            "name": choice["name"],
            "holdout": score(held),
            "versus_constant": compare(held, baseline),
            "versus_original": compare(held, original_holdout),
            "calibration": fit_records[choice["name"]],
            "reliability": backtest.metrics(held)["calibration"],
        }
        store.write_parquet(frame, output / f"{target}_predictions.parquet")
    store.write_parquet(features, output / "features.parquet")
    report = {
        "selection": locked,
        "selected": selected,
        "constant_holdout": score(baseline),
        "original_holdout": score(original_holdout),
        "development": development,
        "holdout_ablations": holdouts,
        "coverage": fixtures.groupby("season").size().to_dict(),
        "caveats": [
            "2025 was inspected before this experiment, not pristine.",
            "Historical xG may have been retrospectively revised.",
            "Goalie usage mixtures are inferred, not confirmed starters.",
            "Regulation xG is prorated; period-specific shots are unavailable.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return {
        k: report[k]
        for k in (
            "selected",
            "constant_holdout",
            "original_holdout",
            "coverage",
            "caveats",
        )
    }
