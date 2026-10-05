"""The daily run: ingest, grade, rate, project, price, decide, publish. Each
stage is also a CLI command so a piece can be rerun on its own."""

import hashlib
import json
import os
import platform
import time
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from backend import backtest, grading, recommendations
from backend.config import (
    BACKTEST_SEASONS,
    HISTORY_START_SEASON,
    SITE_TIME_ZONE,
    STATIC_DIR,
)
from backend.etl import moneypuck, nhl_api, store
from backend.features.windows import venue_windows
from backend.model import goal_map
from backend.model.projections import project_games
from backend.model.ratings import ratings_from_windows, season_of
from backend.odds import partner, verification

SCHEDULE_DAYS = 7
RESULT_LOOKBACK_DAYS = 3
MONEYPUCK_REUSE_HOURS = 6
ODDS_POLL_SECONDS = 300


def site_today() -> date:
    return datetime.now(ZoneInfo(SITE_TIME_ZONE)).date()


def _seasons(day: date) -> range:
    return range(HISTORY_START_SEASON, season_of(day) + 1)


def ingest_moneypuck(day: date, force: bool = False) -> dict:
    """One download per run; a receipt from the last few hours is reused."""
    existing = store.read_receipt("moneypuck")
    if existing and not force:
        observed = datetime.fromisoformat(
            existing["observed_at"].replace("Z", "+00:00")
        )
        # Before opening night the current season has no rows yet; the
        # previous season is what a window needs.
        cached = store.raw_path("moneypuck", f"{season_of(day) - 1}.parquet").exists()
        if cached and datetime.now(UTC) - observed < timedelta(
            hours=MONEYPUCK_REUSE_HOURS
        ):
            return existing
    return moneypuck.ingest(_seasons(day))


def load_teams(day: date) -> tuple[pd.DataFrame, dict]:
    standings, receipt = nhl_api.standings(day)
    colors = pd.read_csv(STATIC_DIR / "teams.csv")
    teams = standings.merge(colors, on="team_abbr", how="left")
    return teams, receipt


def fetch_schedule(day: date) -> tuple[list[dict], dict]:
    rows, receipt = nhl_api.schedule(day)
    end = day + timedelta(days=SCHEDULE_DAYS)
    return [r for r in rows if day <= r["game_date"] < end], receipt


def fetch_results(
    day: date, teams: pd.DataFrame | None = None, pending: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Recent results plus every unresolved decision date, fetched once per day."""
    if teams is None:
        teams, _ = nhl_api.standings(day)
    names = teams.set_index("team_abbr")["team"].to_dict()
    score_rows, receipt = [], None
    dates = {day - timedelta(days=n) for n in range(RESULT_LOOKBACK_DAYS + 1)}
    if pending is not None and not pending.empty:
        starts = pd.to_datetime(pending["start_date"], utc=True, errors="raise")
        dates.update(
            d for d in starts.dt.tz_convert(SITE_TIME_ZONE).dt.date if d <= day
        )
    for result_day in sorted(dates):
        rows, receipt = nhl_api.scores(result_day)
        # The score endpoint has nicknames, unlike the schedule's full names.
        # Resolve identities by abbreviation before the settlement name check.
        for row in rows:
            for side in ("home", "away"):
                abbr = row[f"{side}_abbr"]
                name = names.get(abbr)
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"Missing canonical NHL team name for {abbr}")
                row[f"{side}_team"] = name
        score_rows.extend(rows)
    score_rows = list({row["game_id"]: row for row in score_rows}.values())
    fetched_at = receipt["observed_at"]
    finals = grading.results(score_rows, fetched_at)
    observation = grading.schedule_observation(score_rows, finals, fetched_at)
    return finals, observation, receipt


def grade_pending(day: date, engine) -> dict:
    """Commit official settlements independently of pregame source availability."""
    from backend import publish

    pending = publish.pending_decisions(engine)
    finals, observation, receipt = fetch_results(day, pending=pending)
    settlements = recommendations.settle(
        pending, finals, observation, receipt["observed_at"]
    )
    counts = publish.publish_day(
        engine,
        day,
        results=finals,
        schedule_observation=observation,
        settlements=settlements,
    )
    return {
        **counts,
        "pending_before": len(pending),
        "unresolved": len(pending) - len(settlements),
    }


def build_ratings(day: date, teams: pd.DataFrame):
    games = moneypuck.read_games(_seasons(day))
    windows = venue_windows(games[games.season.ge(season_of(day) - 1)], day)
    coefficients = goal_map.load()
    ratings, league = ratings_from_windows(windows, day, coefficients)
    ratings.attrs["replay_windows"] = windows.to_json(
        orient="table", date_format="iso", double_precision=15
    )
    ratings.attrs["goal_map"] = coefficients
    names = dict(zip(teams["team_abbr"], teams["team"]))
    ratings["team"] = ratings["team_abbr"].map(names).fillna(ratings["team_abbr"])
    return ratings, league


def fetch_offers(
    slate: date | None = None, wait_minutes: float = 0
) -> tuple[pd.DataFrame, list[dict]]:
    """Both partner feeds. The feed keeps serving the previous slate until
    about noon Eastern, so a run that must price `slate` polls for up to
    `wait_minutes` and then fails instead of freezing every game as no_offer."""
    deadline = time.monotonic() + wait_minutes * 60
    while True:
        fetched = [nhl_api.partner_odds(c) for c in nhl_api.PARTNER_COUNTRIES]
        stale = [
            f"{feed['provider']} {feed['odds_date']}"
            for feed, _ in fetched
            if slate is not None and feed["odds_date"] != str(slate)
        ]
        if not stale:
            break
        if time.monotonic() >= deadline:
            raise ValueError(
                f"Partner feed is not on the {slate} slate: {', '.join(stale)}"
            )
        time.sleep(ODDS_POLL_SECONDS)
    frames = [partner.offers(feed) for feed, _ in fetched]
    receipts = [receipt for _, receipt in fetched]
    return pd.concat(frames, ignore_index=True), receipts


def run_backtest() -> pd.DataFrame:
    games = moneypuck.read_games(range(HISTORY_START_SEASON, max(BACKTEST_SEASONS) + 1))
    frame = backtest.run(games, BACKTEST_SEASONS)
    store.write_processed(frame, "backtest.parquet")
    return frame


def daily(
    day: date, engine=None, force_download: bool = False, odds_wait_minutes: float = 0
) -> dict:
    """Settle first, then publish the pregame forecast and decisions atomically."""
    graded = grade_pending(day, engine) if engine is not None else None
    receipts = {"moneypuck": ingest_moneypuck(day, force_download)}
    teams, receipts["standings"] = load_teams(day)
    schedule, receipts["schedule"] = fetch_schedule(day)
    # Only a live run with games today depends on the feed's slate; a rerun of
    # a past day or an off day still grades and publishes ratings.
    live_slate = day == site_today() and any(r["game_date"] == day for r in schedule)
    offers, odds_receipts = fetch_offers(day if live_slate else None, odds_wait_minutes)
    offers, quote_verification = verification.verify(offers, schedule)
    # A manifest preserves both provider receipts under the ledger's single
    # odds source, regardless of which provider supplies the chosen offer.
    receipts["odds"] = store.receipt(
        "partner_odds", json.dumps(odds_receipts, sort_keys=True).encode()
    )
    ratings, league = build_ratings(day, teams)
    as_of = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    projections = project_games(schedule, ratings, league, as_of)
    snapshot = partner.market_snapshot(offers)
    today = projections[projections["game_date"].eq(day)]
    decision_receipts = {
        key: {
            "sha256": receipts[key]["sha256"],
            "observed_at": receipts[key]["observed_at"],
        }
        for key in ("moneypuck", "schedule", "odds")
    }
    decisions = recommendations.decide(today, offers, decision_receipts, as_of)
    # Compact sufficient inputs preserve the exact daily model without retaining
    # the full multi-season MoneyPuck download on every run.
    store.receipt(
        "forecast_replay",
        json.dumps(
            {
                "version": 1,
                "code_revision": os.getenv("GITHUB_SHA"),
                "source_hashes": {
                    str(path.relative_to(Path(__file__).parent)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in sorted(Path(__file__).parent.rglob("*.py"))
                    if "data" not in path.relative_to(Path(__file__).parent).parts
                },
                "runtime": {
                    "python": platform.python_version(),
                    **{
                        package: version(package)
                        for package in ("numpy", "pandas", "scipy")
                    },
                },
                "as_of": as_of,
                "day": str(day),
                "model_version": str(ratings.model_version.iloc[0]),
                "windows": ratings.attrs["replay_windows"],
                "goal_map": ratings.attrs["goal_map"],
                "teams": teams.to_json(
                    orient="table", date_format="iso", double_precision=15
                ),
                "schedule": json.loads(json.dumps(schedule, default=str)),
                "offers": offers.to_json(
                    orient="table", date_format="iso", double_precision=15
                ),
                "receipts": decision_receipts,
            },
            sort_keys=True,
            allow_nan=False,
        ).encode(),
    )
    ratings.attrs.clear()
    store.write_processed(ratings, "ratings", f"{day}.parquet")
    store.write_processed(projections, "projections", f"{day}.parquet")
    store.write_processed(decisions, "decisions", f"{day}.parquet")
    summary = {
        "date": str(day),
        "teams": len(teams),
        "ratings": len(ratings),
        "insufficient": int(ratings["insufficient_window"].sum()),
        "projections": len(projections),
        "today": len(today),
        "offers": len(offers),
        "grading": graded,
        "recommended": int(decisions["status"].eq("recommended").sum()),
        "no_play": int(decisions["status"].eq("no_play").sum()),
        "quote_verification": quote_verification,
    }
    if engine is None:
        return summary
    from backend import publish

    summary["published"] = publish.publish_day(
        engine,
        day,
        teams=teams,
        ratings=ratings,
        projections=projections,
        market_snapshot=snapshot,
        recommendations=decisions,
    )
    return summary
