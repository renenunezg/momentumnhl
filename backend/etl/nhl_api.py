"""NHL API client: schedule, scores, standings, and the partner sportsbook
feed. Every fetch also writes a receipt so decisions can cite their inputs.
The API needs a browser-like User-Agent and its "/now" endpoints redirect to
the dated form, which requests follows."""

from datetime import UTC, date, datetime

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.etl import store

BASE_URL = "https://api-web.nhle.com/v1"
USER_AGENT = "Mozilla/5.0 (momentumnhl; renenunez.dev)"
REQUEST_TIMEOUT_SECONDS = 30
REGULAR_SEASON = 2
FINAL_STATES = ("OFF", "FINAL")
PARTNER_COUNTRIES = ("US", "CA")

_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504)
        )
    ),
)


def _get(path: str, receipt_name: str) -> tuple[dict, dict]:
    response = _session.get(
        f"{BASE_URL}/{path}",
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    fetched_at = datetime.now(UTC)
    return response.json(), store.receipt(receipt_name, response.content, fetched_at)


def season_start_year(season_id: int) -> int:
    return int(season_id) // 10000


def _team_name(team: dict) -> str:
    place = team.get("placeName", {}).get("default", "")
    common = team.get("commonName", {}).get("default", "")
    return f"{place} {common}".strip()


def _game_row(game: dict) -> dict:
    home, away = game["homeTeam"], game["awayTeam"]
    return {
        "game_id": str(game["id"]),
        "season": season_start_year(game["season"]),
        "game_type": int(game["gameType"]),
        "game_date": (
            date.fromisoformat(game["gameDate"]) if "gameDate" in game else None
        ),
        "start_date": game["startTimeUTC"],
        "game_state": game.get("gameState"),
        "schedule_state": game.get("gameScheduleState"),
        "neutral_site": bool(game.get("neutralSite", False)),
        "home_abbr": home["abbrev"],
        "away_abbr": away["abbrev"],
        "home_team": _team_name(home),
        "away_team": _team_name(away),
        "home_logo": home.get("logo"),
        "home_logo_dark": home.get("darkLogo"),
        "away_logo": away.get("logo"),
        "away_logo_dark": away.get("darkLogo"),
        "home_score": home.get("score"),
        "away_score": away.get("score"),
        "last_period_type": (game.get("gameOutcome") or {}).get("lastPeriodType"),
    }


def schedule(start: date) -> tuple[list[dict], dict]:
    """Seven days of regular-season games starting at `start`. The model is
    regular season only, like the sheet's Natural Stat Trick tables."""
    payload, receipt = _get(f"schedule/{start.isoformat()}", f"schedule_{start}")
    rows = []
    for day in payload.get("gameWeek", []):
        for game in day.get("games", []):
            if int(game["gameType"]) != REGULAR_SEASON:
                continue
            row = _game_row(game)
            row["game_date"] = date.fromisoformat(day["date"])
            rows.append(row)
    return rows, receipt


def scores(day: date) -> tuple[list[dict], dict]:
    """That day's games with scores, states, and the finishing period type."""
    payload, receipt = _get(f"score/{day.isoformat()}", f"score_{day}")
    rows = []
    for game in payload.get("games", []):
        if int(game["gameType"]) != REGULAR_SEASON:
            continue
        row = _game_row(game)
        row["game_date"] = date.fromisoformat(game.get("gameDate", day.isoformat()))
        rows.append(row)
    return rows, receipt


def club_season(team_abbr: str, season: int) -> tuple[list[dict], dict]:
    """Every game of one team's season with final scores and the finishing
    period type; 32 calls cover a season for the backtest."""
    season_id = f"{season}{season + 1}"
    payload, receipt = _get(
        f"club-schedule-season/{team_abbr}/{season_id}",
        f"club_season_{team_abbr}_{season}",
    )
    rows = []
    for game in payload.get("games", []):
        if int(game["gameType"]) != REGULAR_SEASON:
            continue
        rows.append(_game_row(game))
    return rows, receipt


def standings(day: date) -> tuple[pd.DataFrame, dict]:
    """Team identity from the standings table. Before opening night the dated
    table is empty, so the current one (last season's final) stands in."""
    payload, receipt = _get(f"standings/{day.isoformat()}", f"standings_{day}")
    if not payload.get("standings"):
        payload, receipt = _get("standings/now", "standings_now")
    rows = [
        {
            "team_abbr": team["teamAbbrev"]["default"],
            "team": team["teamName"]["default"],
            "conference": team.get("conferenceName"),
            "division": team.get("divisionName"),
            "logo_light": team.get("teamLogo"),
            "logo_dark": team.get("teamLogoDark"),
        }
        for team in payload.get("standings", [])
    ]
    return pd.DataFrame(rows).sort_values("team_abbr").reset_index(drop=True), receipt


def partner_odds(country: str) -> tuple[dict, dict]:
    """The partner sportsbook's current slate: US is DraftKings, CA is FanDuel."""
    payload, receipt = _get(f"partner-game/{country}/now", f"partner_odds_{country}")
    partner = payload.get("bettingPartner", {})
    feed = {
        "country": country,
        "provider": partner.get("name"),
        "provider_key": (partner.get("name") or "").lower().replace(" ", ""),
        "last_update": payload.get("lastUpdatedUTC"),
        "odds_date": payload.get("currentOddsDate"),
        "fetched_at": receipt["observed_at"],
        "games": [
            {
                "game_id": str(game["gameId"]),
                "game_type": int(game.get("gameType", REGULAR_SEASON)),
                "start_date": game.get("startTimeUTC"),
                "home_abbr": game["homeTeam"]["abbrev"],
                "away_abbr": game["awayTeam"]["abbrev"],
                "home": game["homeTeam"].get("odds", []),
                "away": game["awayTeam"].get("odds", []),
            }
            for game in payload.get("games", [])
        ],
    }
    return feed, receipt
