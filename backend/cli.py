"""momentumnhl command line. Heavy imports stay inside each subcommand so
startup is instant and a broken optional dependency only breaks its own
command."""

import argparse
import json
from datetime import date

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
    sub.add_parser("fit-goal-map", help="refit the goal map on the configured seasons")
    for name, help_text in (
        ("ratings", "print today's ratings"),
        ("project", "print projections for the next week"),
        ("decide", "print decisions for the day without publishing"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--date")
    sub.add_parser("odds", help="print current partner offers")
    sub.add_parser("grade", help="publish finals and settle pending decisions")
    backtest_parser = sub.add_parser("backtest", help="walk-forward backtest")
    backtest_parser.add_argument("--publish", action="store_true")
    daily = sub.add_parser("daily", help="the production morning run")
    daily.add_argument("--date")
    daily.add_argument("--no-publish", action="store_true")
    daily.add_argument("--force-download", action="store_true")

    args = parser.parse_args()
    if args.command == "ingest":
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
    elif args.command == "odds":
        from backend import pipeline

        offers, _ = pipeline.fetch_offers()
        print(offers.to_string(index=False))
    elif args.command == "decide":
        from backend import pipeline

        summary = pipeline.daily(_day(args.date))
        print(json.dumps(summary, indent=1))
    elif args.command == "grade":
        from backend import pipeline, publish, recommendations

        engine = _engine()
        day = _day(None)
        finals, observation, receipt = pipeline.fetch_results(day)
        pending = publish.pending_decisions(engine)
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
        print(json.dumps(counts))
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
        summary = pipeline.daily(_day(args.date), engine, args.force_download)
        print(json.dumps(summary, indent=1))
