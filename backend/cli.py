"""momentumnhl command line. Heavy imports stay inside each subcommand so
startup is instant and a broken optional dependency only breaks its own
command."""

import argparse
import json
from datetime import date
from pathlib import Path

from backend.config import BACKTEST_SEASONS, HISTORY_START_SEASON


def _engine():
    from backend import db

    return db.engine


def _day(value: str | None) -> date:
    from backend.pipeline import site_today

    return date.fromisoformat(value) if value else site_today()


def main() -> None:
    parser = argparse.ArgumentParser(prog="backend")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="download MoneyPuck and cache by season")
    ingest.add_argument("--seasons", nargs="+", type=int)
    goalie_ingest = sub.add_parser(
        "ingest-goalies", help="cache listed goalie game logs"
    )
    goalie_ingest.add_argument("--seasons", nargs="+", type=int, required=True)
    sub.add_parser("fit-goal-map", help="refit the goal map on the configured seasons")
    sub.add_parser(
        "ingest-play-by-play", help="cache play-by-play for every backtest game"
    )
    sub.add_parser("ingame-backtest", help="chronological in-game win probability")
    live = sub.add_parser(
        "live-win-probability", help="score the games in progress; read only by default"
    )
    live.add_argument("--watch", action="store_true")
    live.add_argument("--interval", type=int, default=30)
    live.add_argument("--duration", type=int)
    live.add_argument("--publish", action="store_true")
    for name, help_text in (
        ("ratings", "print today's ratings"),
        ("project", "print projections for the next week"),
        ("decide", "print decisions for the day without publishing"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--date")
    sub.add_parser("odds", help="print current partner offers")
    verify = sub.add_parser("verify-quotes", help="independently check partner prices")
    verify.add_argument("--date")
    sub.add_parser("grade", help="publish finals and settle pending decisions")
    backtest_parser = sub.add_parser("backtest", help="walk-forward backtest")
    backtest_parser.add_argument("--publish", action="store_true")
    validation = sub.add_parser("validate", help="chronological candidate evaluation")
    validation.add_argument("--output", type=Path, required=True)
    candidate = sub.add_parser("candidate", help="unpublished candidate forecasts")
    candidate.add_argument("--date")
    candidate.add_argument("--selection", type=Path, required=True)
    candidate.add_argument("--confirmations", type=Path)
    market = sub.add_parser(
        "validate-market", help="timestamped market-blend evaluation"
    )
    market.add_argument("--predictions", type=Path, required=True)
    market.add_argument("--snapshots", type=Path, required=True)
    daily = sub.add_parser("daily", help="the production morning run")
    daily.add_argument("--date")
    daily.add_argument("--no-publish", action="store_true")
    daily.add_argument("--force-download", action="store_true")
    daily.add_argument(
        "--odds-wait-minutes",
        type=float,
        default=0,
        help="how long to poll for the partner feed to reach today's slate",
    )

    args = parser.parse_args()
    if args.command == "validate":
        from backend.validation import run

        print(json.dumps(run(args.output), indent=2))
    elif args.command == "candidate":
        import pandas as pd

        from backend.candidate import project

        confirmations = (
            pd.read_parquet(args.confirmations) if args.confirmations else None
        )
        print(
            project(_day(args.date), args.selection, confirmations).to_json(
                orient="records", date_format="iso", indent=2
            )
        )
    elif args.command == "validate-market":
        import pandas as pd

        from backend.model.market import evaluate

        print(
            json.dumps(
                evaluate(
                    pd.read_parquet(args.predictions), pd.read_parquet(args.snapshots)
                ),
                indent=2,
            )
        )
    elif args.command == "ingest-goalies":
        from backend.etl import goalies

        frame = goalies.load(args.seasons, download=True)
        print(json.dumps({"rows": len(frame), "games": frame.game_id.nunique()}))
    elif args.command == "ingest-play-by-play":
        from backend.etl import play_by_play, store

        games = store.read_processed("backtest.parquet", columns=["game_id", "season"])
        print(json.dumps(play_by_play.ingest(games)))
    elif args.command == "ingame-backtest":
        from backend import ingame_backtest

        report = ingame_backtest.run()
        print(json.dumps({k: report[k] for k in ("data", "selection", "chosen")}))
    elif args.command == "live-win-probability":
        import logging
        import sys
        from datetime import UTC, datetime, timedelta

        from backend import live_publish
        from backend.db import writes_allowed
        from backend.etl import live_feed
        from backend.model import ingame

        if args.interval < 30:
            raise SystemExit("--interval must be at least 30 seconds")
        if args.duration is not None and (args.duration <= 0 or not args.watch):
            raise SystemExit("--duration requires --watch and a positive length")
        if args.publish and not writes_allowed():
            raise SystemExit("publication requires MOMENTUMNHL_DB_WRITES=1")
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
        publisher = live_publish.LivePublisher(
            ingame.load(),
            load_games=live_publish.load_games,
            load_saved=live_publish.load_saved,
            fetch_states=live_feed.fetch_states,
            write=live_publish.write_snapshots if args.publish else None,
            expires_at=(
                datetime.now(UTC) + timedelta(seconds=args.duration)
                if args.duration
                else None
            ),
        )
        live_publish.run(
            publisher, watch=args.watch, interval=args.interval, duration=args.duration
        )
    elif args.command == "ingest":
        from backend.etl import moneypuck

        seasons = args.seasons or range(HISTORY_START_SEASON, date.today().year + 1)
        print(json.dumps(moneypuck.ingest(seasons)))
    elif args.command == "fit-goal-map":
        from backend.etl import moneypuck
        from backend.model import goal_map

        fitted = goal_map.fit(moneypuck.read_games(range(HISTORY_START_SEASON, 2100)))
        goal_map.save(fitted)
        print(json.dumps({k: fitted[k] for k in ("rows", "seasons")}))
        for side in ("for", "against"):
            print(side, "r_squared", round(fitted[side]["r_squared"], 4))
    elif args.command == "ratings":
        from backend import pipeline

        day = _day(args.date)
        teams, _ = pipeline.load_teams(day)
        ratings, league = pipeline.build_ratings(day, teams)
        print(json.dumps({k: round(v, 3) for k, v in league.items()}))
        columns = [
            "team_abbr",
            "rating",
            "home_xgf",
            "home_xga",
            "away_xgf",
            "away_xga",
        ]
        print(ratings[columns].round(3).to_string(index=False))
    elif args.command == "project":
        from backend import pipeline
        from backend.model.projections import project_games

        day = _day(args.date)
        teams, _ = pipeline.load_teams(day)
        ratings, league = pipeline.build_ratings(day, teams)
        schedule, _ = pipeline.fetch_schedule(day)
        frame = project_games(schedule, ratings, league, day.isoformat())
        columns = [
            "game_date", "away_team_abbr", "home_team_abbr", "away_lambda",
            "home_lambda", "home_win_prob", "model_total", "home_fair_price",
        ]  # fmt: skip
        print(frame[columns].round(3).to_string(index=False))
    elif args.command == "verify-quotes":
        from backend import pipeline
        from backend.odds import verification

        schedule, _ = pipeline.fetch_schedule(_day(args.date))
        offers, _ = pipeline.fetch_offers()
        _, report = verification.verify(offers, schedule)
        print(json.dumps(report, indent=2))
    elif args.command == "odds":
        from backend import pipeline

        offers, _ = pipeline.fetch_offers()
        print(offers.to_string(index=False))
    elif args.command == "decide":
        from backend import pipeline

        summary = pipeline.daily(_day(args.date))
        print(json.dumps(summary, indent=1))
    elif args.command == "grade":
        from backend import pipeline

        print(json.dumps(pipeline.grade_pending(_day(None), _engine())))
    elif args.command == "backtest":
        from backend import backtest, pipeline

        frame = pipeline.run_backtest()
        metrics = backtest.metrics(frame)
        print(json.dumps({k: v for k, v in metrics.items() if k != "calibration"}))
        if args.publish:
            from backend import publish

            print("published", publish.publish_backtest(_engine(), frame))
        else:
            print("seasons", list(BACKTEST_SEASONS))
    elif args.command == "daily":
        from backend import pipeline

        engine = None if args.no_publish else _engine()
        summary = pipeline.daily(
            _day(args.date), engine, args.force_download, args.odds_wait_minutes
        )
        print(json.dumps(summary, indent=1))
