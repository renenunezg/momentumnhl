"""Live game states from the NHL score feed, one call per slate date."""

from dataclasses import dataclass
from datetime import date

from backend.etl import nhl_api
from backend.etl.nhl_api import FINAL_STATES

LIVE_STATES = ("LIVE", "CRIT")
CALLED_OFF = ("PPD", "CNCL")
PERIOD_SECONDS = 1200


@dataclass(frozen=True, slots=True)
class LiveState:
    """What the feed knows about one game; None means it did not say."""

    status: str  # scheduled, live, final, or off
    home_score: int = 0
    away_score: int = 0
    period: int | None = None
    period_type: str | None = None  # REG, OT, or SO
    # Left in the period, or zero once it has ended.
    clock_seconds: int | None = None
    intermission: bool = False
    # Away goalie, away skaters, home skaters, home goalie; None at full
    # strength, when the feed carries no situation.
    situation_code: str | None = None
    advantage_seconds: int | None = None


def _whole(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value) if value == int(value) and value >= 0 else None


def parse_game(game: dict) -> LiveState:
    if game.get("gameScheduleState") in CALLED_OFF:
        return LiveState("off")
    state = game["gameState"]
    if state not in FINAL_STATES + LIVE_STATES:
        return LiveState("scheduled")
    home = _whole(game["homeTeam"].get("score"))
    away = _whole(game["awayTeam"].get("score"))
    if home is None or away is None:
        raise ValueError(f"Missing or invalid score for game {game['id']}")
    if state in FINAL_STATES:
        return LiveState("final", home, away)
    period = game.get("periodDescriptor") or {}
    clock = game.get("clock") or {}
    situation = game.get("situation") or {}
    intermission = bool(clock.get("inIntermission"))
    return LiveState(
        status="live",
        home_score=home,
        away_score=away,
        period=_whole(period.get("number")) or None,
        period_type=period.get("periodType"),
        # During a break the feed's clock counts the intermission down.
        clock_seconds=0 if intermission else _whole(clock.get("secondsRemaining")),
        intermission=intermission,
        situation_code=situation.get("situationCode"),
        advantage_seconds=_whole(situation.get("secondsRemaining")),
    )


def fetch_states(days: set[date]) -> dict[str, LiveState]:
    """Every game on the given slate dates, keyed by NHL game id."""
    return {
        str(game["id"]): parse_game(game)
        for day in sorted(days)
        for game in nhl_api.live_scores(day)
    }
