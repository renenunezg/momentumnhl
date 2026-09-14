"""The sheet's regression block: goals per 60 as a linear function of ten
shot-quality features, one fit for goals for and one for goals against. The
workbook hard-coded coefficients fitted on Natural Stat Trick data; the same
fit on MoneyPuck team-seasons lives in backend/data_static/goal_map.json and
is applied the sheet's way, to every situation and venue alike."""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from backend.config import GOAL_MAP_SEASONS, STATIC_DIR
from backend.etl.moneypuck import COUNT_COLUMNS
from backend.features.windows import AGAINST_FEATURES, FOR_FEATURES, feature_rates

GOAL_MAP_PATH = STATIC_DIR / "goal_map.json"
VERSION = "goal-map-v1"


def _ols(frame: pd.DataFrame, features: list[str], target: str) -> dict:
    x = np.column_stack([np.ones(len(frame)), frame[features].to_numpy(dtype=float)])
    y = frame[target].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residual = float(((y - fitted) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    return {
        "intercept": float(beta[0]),
        "coefficients": dict(zip(features, map(float, beta[1:]))),
        "r_squared": 1 - residual / total if total else float("nan"),
    }


def fit(games: pd.DataFrame, seasons=GOAL_MAP_SEASONS) -> dict:
    """Team-season even-strength aggregates, all venues pooled."""
    rows = games[games["season"].isin(seasons) & games["situation"].eq("ev")]
    summed = rows.groupby(["team_abbr", "season"], as_index=False)[COUNT_COLUMNS].sum()
    summed["situation"] = "ev"
    rates = feature_rates(summed)
    return {
        "version": VERSION,
        "seasons": list(seasons),
        "rows": int(len(rates)),
        "for": _ols(rates, FOR_FEATURES, "gf60"),
        "against": _ols(rates, AGAINST_FEATURES, "ga60"),
        "fitted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def save(goal_map: dict) -> None:
    GOAL_MAP_PATH.write_text(json.dumps(goal_map, indent=1) + "\n")


def load() -> dict:
    return json.loads(GOAL_MAP_PATH.read_text())


def predict_per60(features: pd.DataFrame, side: str, goal_map: dict) -> pd.Series:
    """Expected goals per 60 for every row, 'for' or 'against'."""
    spec = goal_map[side]
    names = list(spec["coefficients"])
    coefficients = np.array([spec["coefficients"][name] for name in names])
    values = features[names].to_numpy(dtype=float) @ coefficients + spec["intercept"]
    return pd.Series(values, index=features.index)
