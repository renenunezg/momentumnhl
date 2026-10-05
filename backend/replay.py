"""Offline reconstruction from one retained forecast_replay source body."""

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
from backend.model.projections import project_games
from backend.model.ratings import ratings_from_windows


def forecast(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payload = json.loads(gzip.decompress(path.read_bytes()))
    if payload["version"] != 1:
        raise ValueError("Unsupported forecast replay version")
    runtime = {
        "python": platform.python_version(),
        **{package: version(package) for package in ("numpy", "pandas", "scipy")},
    }
    if payload["runtime"] != runtime:
        raise ValueError("Replay requires the recorded runtime versions")
    root = Path(__file__).parent.resolve()
    for name, digest in payload["source_hashes"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
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
    projections = project_games(schedule, ratings, league, payload["as_of"])
    offers = pd.read_json(io.StringIO(payload["offers"]), orient="table")
    decisions = recommendations.decide(
        projections[projections.game_date.eq(day)],
        offers,
        payload["receipts"],
        payload["as_of"],
    )
    return ratings, projections, decisions
