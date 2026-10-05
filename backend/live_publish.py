"""Publish live win probabilities across the Supabase boundary.

One score-feed call covers a slate date. None is made before a tracked
game's scheduled puck drop or after every tracked game is terminal, and the
worker exits as soon as nothing is left to watch. Each state is scored by the
in-game model against the pregame goal rates in nhl.game_projections, which
the database freezes at puck drop; pregame tables are never written here.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np

from backend.etl.live_feed import PERIOD_SECONDS, LiveState
from backend.model import ingame
from backend.publish import SCHEMA

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PROBABILITY_SOURCE = "pregame_anchored_game_state"
TERMINAL_STATES = {"Final", "Off"}
FULL_STRENGTH = "1551"
# The dispatcher treats a row newer than four minutes as a living worker.
HEARTBEAT_SECONDS = 120
SCHEDULE_REFRESH_SECONDS = 300
START_LEAD = timedelta(minutes=15)
# Matches the dispatcher window, so a recovered worker resumes the same games.
START_LOOKBACK = timedelta(hours=6)
# A game still unstarted this long after puck drop is abandoned, not polled.
NO_START_LIMIT = timedelta(hours=3)
MAX_BACKOFF_SECONDS = 300
VOLATILE_FIELDS = {"fetched_at", "worker_expires_at"}


@dataclass(frozen=True, slots=True)
class Game:
    game_id: str
    season: int
    game_date: date
    start: datetime
    home_team: str
    away_team: str
    home_lambda: float
    away_lambda: float
    as_of: datetime
    model_version: str

    def anchor(self) -> dict:
        return {
            "home_lambda": self.home_lambda,
            "away_lambda": self.away_lambda,
            "as_of": self.as_of.isoformat(),
            "model_version": self.model_version,
        }


def _elapsed(state: LiveState) -> int | None:
    """Seconds of regulation played, or None when the feed's clock is unusable."""
    if state.period_type in ("OT", "SO"):
        return ingame.REGULATION_SECONDS
    clock = state.clock_seconds
    if (
        state.period_type != "REG"
        or not state.period
        or clock is None
        or not 0 <= clock <= PERIOD_SECONDS
    ):
        return None
    return min(state.period * PERIOD_SECONDS - clock, ingame.REGULATION_SECONDS)


def _probability(game: Game, params: dict, state: LiveState, elapsed: int) -> float:
    margin = state.home_score - state.away_score
    # A tie at the horn is overtime whether or not the feed has moved on.
    overtime = state.period_type != "REG" or (
        elapsed == ingame.REGULATION_SECONDS and margin == 0
    )
    if overtime and margin:
        # Sudden death: the goal is on the board before the game goes final.
        return float(margin > 0)
    # A shootout is the same coin flip whatever the feed says of manpower.
    code = (
        FULL_STRENGTH
        if state.period_type == "SO"
        else state.situation_code or FULL_STRENGTH
    )
    return float(
        ingame.home_win_probability(
            params,
            game=np.zeros(1, dtype=int),
            margin=[margin],
            seconds_remaining=[ingame.REGULATION_SECONDS - elapsed],
            overtime=[overtime],
            kind=ingame.kinds([code], [elapsed], [margin], [overtime]),
            advantage_seconds=[state.advantage_seconds or 0],
            home_lambda=[game.home_lambda],
            away_lambda=[game.away_lambda],
        )[0]
    )


def _point(elapsed: int, home: int, away: int, probability: float) -> dict:
    return {"s": elapsed, "h": home, "a": away, "p": round(probability, 4)}


def score_game(
    game: Game,
    state: LiveState | None,
    params: dict,
    saved: dict | None,
    now: datetime,
    *,
    board_fetched: bool,
) -> dict:
    """One game's serving payload from its live state and pregame anchor."""
    opening = _probability(game, params, LiveState("live", period_type="REG"), 0)
    # The anchor can still be republished until puck drop, so the opening
    # point always follows it.
    history = [_point(0, 0, 0, opening), *((saved or {}).get("history") or [])[1:]]
    last = history[-1]["p"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "game_id": game.game_id,
        "season": game.season,
        "abstract_state": "Pre",
        "status": "scheduled",
        "home_team": game.home_team,
        "away_team": game.away_team,
        "fetched_at": now.isoformat(),
        "probability_source": PROBABILITY_SOURCE,
        "model_version": params["version"],
        "anchor": game.anchor(),
        "home_win_probability": last,
        "away_win_probability": 1 - last,
        "history": history,
    }

    def unavailable(reason: str) -> dict:
        result.update(
            home_win_probability=None,
            away_win_probability=None,
            unavailable_reason=reason,
        )
        return result

    if not board_fetched:
        return result
    if state is not None and state.status == "off":
        result.update(abstract_state="Off", status="off")
        return unavailable("Game was called off")
    if state is None or state.status == "scheduled":
        if now - game.start > NO_START_LIMIT:
            result["abstract_state"] = "Off"
            return unavailable("Game did not start")
        return result
    home, away = state.home_score, state.away_score
    # An NHL final has a winner; a tied one is a feed still catching up on
    # the deciding goal, so the game stays live until it does.
    final = state.status == "final" and home != away
    result.update(
        abstract_state="Final" if final else "Live",
        status="final" if final else "live",
        home_score=home,
        away_score=away,
    )
    if final:
        probability, elapsed = float(home > away), ingame.REGULATION_SECONDS
    elif state.status == "final":
        return result
    else:
        elapsed = _elapsed(state)
        if elapsed is None:
            return unavailable("Score feed is missing the period or clock")
        probability = _probability(game, params, state, elapsed)
        clock = state.clock_seconds
        result.update(
            period=state.period,
            period_type=state.period_type,
            clock=None if clock is None else f"{clock // 60}:{clock % 60:02d}",
            intermission=state.intermission,
        )
    result.update(
        home_win_probability=probability, away_win_probability=1 - probability
    )
    point = _point(elapsed, home, away, probability)
    if history[-1] != point:
        result["history"] = [*history, point]
    return result


def _signature(payload: dict) -> str:
    content = {k: v for k, v in payload.items() if k not in VOLATILE_FIELDS}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class LivePublisher:
    """Tracks the games in the puck-drop window and writes only what changed."""

    def __init__(
        self,
        params: dict,
        *,
        load_games,
        load_saved,
        fetch_states,
        write=None,
        expires_at: datetime | None = None,
    ):
        self.params = params
        self.load_games = load_games
        self.load_saved = load_saved
        self.fetch_states = fetch_states
        self.write = write
        self.expires_at = expires_at.isoformat() if expires_at else None
        self.games: dict[str, Game] = {}
        self.payloads: dict[str, dict] = {}
        self.written: dict[str, tuple[str, datetime]] = {}
        self.done: set[str] = set()
        self.next_schedule: datetime | None = None

    def _refresh_schedule(self, now: datetime) -> None:
        if self.next_schedule and now < self.next_schedule:
            return
        window = {
            game.game_id: game
            for game in self.load_games(now - START_LOOKBACK, now + START_LEAD)
        }
        unseen = [game_id for game_id in window if game_id not in self.games]
        saved = self.load_saved(unseen) if unseen and self.write else {}
        for game_id in unseen:
            payload = saved.get(game_id)
            if payload:
                self.payloads[game_id] = payload
                if payload["abstract_state"] in TERMINAL_STATES:
                    self.done.add(game_id)
        for game_id in set(self.games) - set(window):
            self.payloads.pop(game_id, None)
            self.written.pop(game_id, None)
            self.done.discard(game_id)
        # Every refresh rereads the projections, so a forecast republished
        # before puck drop is the one a running worker scores against.
        self.games = window
        self.next_schedule = now + timedelta(seconds=SCHEDULE_REFRESH_SECONDS)

    def poll(self, now: datetime) -> bool:
        """Score the open games once; False when nothing is left to watch."""
        self._refresh_schedule(now)
        open_games = [g for g in self.games.values() if g.game_id not in self.done]
        if not open_games:
            return False
        # Before the first scheduled puck drop the database alone is enough.
        started = [game for game in open_games if game.start <= now]
        states = self.fetch_states({g.game_date for g in started}) if started else {}
        changed, finished = [], []
        for game in open_games:
            payload = score_game(
                game,
                states.get(game.game_id),
                self.params,
                self.payloads.get(game.game_id),
                now,
                board_fetched=game.start <= now,
            )
            payload["worker_expires_at"] = self.expires_at
            self.payloads[game.game_id] = payload
            signature = _signature(payload)
            previous = self.written.get(game.game_id)
            if (
                previous
                and previous[0] == signature
                and (now - previous[1]).total_seconds() < HEARTBEAT_SECONDS
            ):
                continue
            changed.append((payload, signature))
            if payload["abstract_state"] in TERMINAL_STATES:
                finished.append(game.game_id)
        if changed and self.write:
            self.write([payload for payload, _ in changed])
        # Acknowledge only after the batch commits, so a failed final retries.
        for payload, signature in changed:
            self.written[payload["game_id"]] = (signature, now)
            log.info(
                "%s %s @ %s: %s, home WP %s%s",
                payload["game_id"],
                payload["away_team"],
                payload["home_team"],
                payload["abstract_state"],
                payload["home_win_probability"],
                " (published)" if self.write else " (read only)",
            )
        self.done.update(finished)
        return len(self.done) < len(self.games)


def run(
    publisher: LivePublisher,
    *,
    watch: bool,
    interval: int,
    duration: int | None = None,
    sleep=time.sleep,
    monotonic=time.monotonic,
    now=lambda: datetime.now(UTC),
) -> None:
    deadline = monotonic() + duration if duration else float("inf")
    failures = 0
    while True:
        started = monotonic()
        try:
            more = publisher.poll(now())
            failures = 0
        except Exception:  # noqa: BLE001 - a watch worker outlives a bad poll
            if not watch:
                raise
            log.exception("Refresh failed; previous snapshots age visibly on the site")
            more, failures = True, failures + 1
        if not more:
            log.info("No live or imminent games; stopping until the next dispatch")
            return
        if not watch or monotonic() >= deadline:
            return
        # A refused or failing feed is asked less often, not hammered.
        wait = min(MAX_BACKOFF_SECONDS, interval * 2**failures)
        sleep(max(1, min(deadline - monotonic(), wait - (monotonic() - started))))
        if monotonic() >= deadline:
            return


def load_games(start: datetime, end: datetime) -> list[Game]:
    from sqlalchemy import text

    from backend.db import engine

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT game_id, season, game_date, start_date, home_team, "
                "away_team, home_lambda, away_lambda, as_of, model_version "
                f"FROM {SCHEMA}.game_projections "
                "WHERE start_date BETWEEN :start AND :end"
            ),
            {"start": start, "end": end},
        ).mappings()
        return [
            Game(
                game_id=row["game_id"],
                season=int(row["season"]),
                game_date=row["game_date"],
                start=row["start_date"],
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_lambda=float(row["home_lambda"]),
                away_lambda=float(row["away_lambda"]),
                as_of=row["as_of"],
                model_version=row["model_version"],
            )
            for row in rows
        ]


def load_saved(game_ids: list[str]) -> dict[str, dict]:
    from sqlalchemy import text

    from backend.db import engine

    with engine.connect() as conn:
        return dict(
            conn.execute(
                text(
                    f"SELECT game_id, payload FROM {SCHEMA}.live_win_probability "
                    "WHERE game_id = ANY(:game_ids)"
                ),
                {"game_ids": game_ids},
            )
            .tuples()
            .all()
        )


def write_snapshots(payloads: list[dict]) -> None:
    from sqlalchemy import text

    from backend.db import engine

    rows = [
        {"game_id": p["game_id"], "updated_at": p["fetched_at"], "payload": p}
        for p in payloads
    ]
    with engine.begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO {SCHEMA}.live_win_probability "
                "(game_id, updated_at, payload) "
                "SELECT game_id, updated_at, payload "
                "FROM jsonb_to_recordset(CAST(:snapshots AS jsonb)) "
                "AS incoming(game_id text, updated_at timestamptz, payload jsonb) "
                "ON CONFLICT (game_id) DO UPDATE SET "
                "updated_at = EXCLUDED.updated_at, payload = EXCLUDED.payload "
                f"WHERE {SCHEMA}.live_win_probability.updated_at "
                "< EXCLUDED.updated_at"
            ),
            {"snapshots": json.dumps(rows, allow_nan=False)},
        )
