"""The forecast_replay contract: the compact inputs a daily run retains, and
the offline reconstruction that reads them back. Both sides live here so the
archive and its reader cannot drift apart."""

import gzip
import hashlib
import io
import json
import platform
from datetime import date
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from backend import recommendations
from backend.etl import store
from backend.model.projections import anchor_to_market, project_games
from backend.model.ratings import ratings_from_windows

VERSION = 1
SOURCE_ROOT = Path(__file__).parent.resolve()
RUNTIME_PACKAGES = ("numpy", "pandas", "scipy")


def _runtime() -> dict:
    return {
        "python": platform.python_version(),
        **{package: version(package) for package in RUNTIME_PACKAGES},
    }


def _table(frame: pd.DataFrame) -> str:
    return frame.to_json(orient="table", date_format="iso", double_precision=15)


def archive(
    *,
    day: date,
    as_of: str,
    code_revision: str | None,
    model_version: str,
    windows: pd.DataFrame,
    goal_map: dict,
    teams: pd.DataFrame,
    schedule: list[dict],
    offers: pd.DataFrame,
    receipts: dict,
) -> dict:
    """Retain sufficient inputs to recompute the day's exact forecast without
    keeping the full multi-season MoneyPuck download on every run."""
    payload = {
        "version": VERSION,
        "code_revision": code_revision,
        "source_hashes": {
            str(path.relative_to(SOURCE_ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(SOURCE_ROOT.rglob("*.py"))
            if "data" not in path.relative_to(SOURCE_ROOT).parts
        },
        "runtime": _runtime(),
        "as_of": as_of,
        "day": str(day),
        "model_version": model_version,
        "windows": _table(windows),
        "goal_map": goal_map,
        "teams": _table(teams),
        "schedule": json.loads(json.dumps(schedule, default=str)),
        "offers": _table(offers),
        "receipts": receipts,
    }
    return store.receipt(
        "forecast_replay",
        json.dumps(payload, sort_keys=True, allow_nan=False).encode(),
    )


def forecast(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Offline reconstruction from one retained forecast_replay source body."""
    payload = json.loads(gzip.decompress(path.read_bytes()))
    if payload["version"] != VERSION:
        raise ValueError("Unsupported forecast replay version")
    if payload["runtime"] != _runtime():
        raise ValueError("Replay requires the recorded runtime versions")
    for name, digest in payload["source_hashes"].items():
        path = (SOURCE_ROOT / name).resolve()
        if not path.is_relative_to(SOURCE_ROOT):
            raise ValueError("Invalid replay source path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Replay requires the recorded source revision: {name}")
    day = date.fromisoformat(payload["day"])
    windows = pd.read_json(io.StringIO(payload["windows"]), orient="table")
    ratings, league = ratings_from_windows(windows, day, payload["goal_map"])
    if not ratings.model_version.eq(payload["model_version"]).all():
        raise ValueError("Replay requires the archived model version")
    teams = pd.read_json(io.StringIO(payload["teams"]), orient="table")
    names = teams.set_index("team_abbr").team.to_dict()
    ratings["team"] = ratings.team_abbr.map(names).fillna(ratings.team_abbr)
    schedule = payload["schedule"]
    for game in schedule:
        game["game_date"] = date.fromisoformat(game["game_date"])
    offers = pd.read_json(io.StringIO(payload["offers"]), orient="table")
    pure = project_games(schedule, ratings, league, payload["as_of"])
    projections = anchor_to_market(
        pure,
        recommendations.market_consensus(pure, offers, payload["as_of"]),
    )
    decisions = recommendations.decide(
        projections[projections.game_date.eq(day)],
        offers,
        payload["receipts"],
        payload["as_of"],
    )
    return ratings, projections, decisions
