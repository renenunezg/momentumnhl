"""Historical closing lines from the local ESPN odds cache, for the backtest's
market benchmark only. The cache is one approved pull that is never refreshed
here and never prices a live game."""

import gzip
import json

import pandas as pd

from backend.config import RAW_DIR
from backend.model.pricing import american_to_decimal

CACHE_DIR = RAW_DIR / "espn_odds"
CLOSING_COLUMNS = [
    "closing_home_price",
    "closing_away_price",
    "closing_total",
    "closing_source",
]
SOURCE_PREFIX = "espn:"
REGULAR_SEASON = 2
# ESPN abbreviations that differ from the NHL's.
_ABBREVIATIONS = {"TB": "TBL", "NJ": "NJD", "SJ": "SJS", "LA": "LAK", "UTAH": "UTA"}
# Several books on the endpoint quote 3-way regulation lines. A 2-way pair's
# implied probabilities sum to just over one.
MAX_TWO_WAY_OVERROUND = 0.10
# The 2022-23 items carry a price where the closing total belongs.
MAX_TOTAL_GOALS = 20
# Pregame books in order of preference; ESPN's own default book of each era
# leads. Live-odds feeds and the remaining books are never a close.
_BOOKS = (
    "DraftKings",
    "ESPN BET",
    "Caesars Sportsbook (New Jersey)",
    "Caesars Sportsbook (Colorado)",
    "Caesars Sportsbook",
    "Caesars Sportsbook (Tennessee)",
    "MGM",
    "PointsBet",
)


def _american(quote) -> float | None:
    text = (quote or {}).get("american")
    if not text:
        return None
    return 100.0 if text.upper() == "EVEN" else float(text)


def _moneyline(item: dict) -> tuple[float, float] | None:
    prices = [
        _american(((item.get(side) or {}).get("close") or {}).get("moneyLine"))
        for side in ("homeTeamOdds", "awayTeamOdds")
    ]
    if None in prices or min(abs(p) for p in prices) < 100:
        return None
    overround = sum(1 / american_to_decimal(p) for p in prices) - 1
    return tuple(prices) if 0 < overround <= MAX_TWO_WAY_OVERROUND else None


def _total(item: dict) -> float | None:
    total = _american((item.get("close") or {}).get("total"))
    valid = total is not None and 0 < total < MAX_TOTAL_GOALS and total * 2 % 1 == 0
    return total if valid else None


def closing_lines() -> pd.DataFrame:
    """One book's close per regular-season game, keyed by date and NHL
    abbreviations: the preferred book with a two-way closing moneyline, and
    that book's closing total when it recorded one."""
    paths = sorted(CACHE_DIR.glob("*.json.gz"))
    if not paths:
        raise FileNotFoundError(f"No ESPN odds cache under {CACHE_DIR}")
    rows = []
    for path in paths:
        with gzip.open(path) as handle:
            day = json.load(handle)
        for event in day["events"]:
            if event["season_type"] != REGULAR_SEASON:
                continue
            closes = {
                item["provider"]["name"]: item
                for item in event["items"]
                if _moneyline(item)
            }
            provider = next((book for book in _BOOKS if book in closes), None)
            if provider is None:
                continue
            item = closes[provider]
            home_price, away_price = _moneyline(item)
            rows.append(
                {
                    "game_date": pd.Timestamp(day["date"]).date(),
                    "home_team": _ABBREVIATIONS.get(event["home"], event["home"]),
                    "away_team": _ABBREVIATIONS.get(event["away"], event["away"]),
                    "closing_home_price": home_price,
                    "closing_away_price": away_price,
                    "closing_total": _total(item),
                    "closing_source": SOURCE_PREFIX + provider,
                }
            )
    return pd.DataFrame(rows)
