"""Provider identity, settlement and offline forecast replay."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from backend import grading as official_grading
from backend import pipeline, publish
from backend.etl import nhl_api
from backend.recommendations import SETTLEMENT_COLUMNS, settle


def test_official_score_feed_preserves_settlement_identity(monkeypatch):
    payload = json.loads(
        (Path(__file__).parent / "fixtures/score_2026_09_29_bos_nyr.json").read_text()
    )
    observed_at = "2026-09-30T14:01:00.402294Z"
    receipt = {"observed_at": observed_at}
    teams = pd.DataFrame(
        {"team_abbr": ["BOS", "NYR"], "team": ["Boston Bruins", "New York Rangers"]}
    )
    monkeypatch.setattr(
        nhl_api,
        "_get",
        lambda path, name: (
            payload if path == "score/2026-09-29" else {"games": []},
            receipt,
        ),
    )
    calls = []

    def standings(day):
        calls.append(day)
        return teams, receipt

    monkeypatch.setattr(nhl_api, "standings", standings)
    day = date(2026, 9, 30)
    # Daily reuses its team source; the standalone grade command fetches it once.
    daily = pipeline.fetch_results(day, teams)
    assert not calls
    grading = pipeline.fetch_results(day)
    assert calls == [day]
    for finals, observation, _ in (daily, grading):
        result = finals.iloc[0]
        fixture = observation.iloc[0]
        assert result.home_team == fixture.home_team == "Boston Bruins"
        assert result.away_team == fixture.away_team == "New York Rangers"
        assert (result.home_goals, result.away_goals) == (3, 0)
        pending = pd.DataFrame(
            [
                {
                    "game_id": result.game_id,
                    "market": market,
                    "status": status,
                    "start_date": result.start_date,
                    "side": "home",
                    "point": None,
                    "price": -110,
                    "stake_units": stake,
                }
                for market, status, stake in (
                    ("h2h", "recommended", 1),
                    ("totals", "no_play", 0),
                )
            ]
        )
        settlements = publish._prepare(
            settle(pending, finals, observation, "2026-09-30T14:01:01.488063Z"),
            SETTLEMENT_COLUMNS,
        )
        saved = publish._prepare(observation, publish.RECOMMENDATION_SCHEDULE_COLUMNS)
        assert settlements.outcome.tolist() == ["win", "no_play"]
        assert settlements.result_source_at.eq(saved.iloc[0].observed_at).all()
        assert saved.iloc[0].completed
    # Recovery follows the unresolved ledger beyond the routine lookback.
    overdue = pending.assign(start_date="2026-09-29T23:00:00Z")
    recovered = pipeline.fetch_results(date(2026, 10, 5), teams, overdue)
    assert len(recovered[0]) == 1
    postponed = nhl_api.scores(date(2026, 9, 29))[0]
    for row in postponed:
        row["schedule_state"] = "PPD"
        row["home_team"], row["away_team"] = "Boston Bruins", "New York Rangers"
    observation = official_grading.schedule_observation(
        postponed, recovered[0].iloc[:0], observed_at
    )
    assert observation.game_status.eq("postponed").all()
    voided = settle(pending, recovered[0].iloc[:0], observation, observed_at)
    assert voided.outcome.tolist() == ["void", "no_play"]

    # Settlement commits even when a later forecast source fails.
    monkeypatch.setattr(publish, "pending_decisions", lambda _: overdue)
    writes = []
    monkeypatch.setattr(
        publish, "publish_day", lambda *a, **kw: writes.append(kw) or {}
    )

    def failed_ingest(*_):
        raise RuntimeError("MoneyPuck unavailable")

    monkeypatch.setattr(pipeline, "ingest_moneypuck", failed_ingest)
    with pytest.raises(RuntimeError, match="MoneyPuck unavailable"):
        pipeline.daily(date(2026, 10, 5), engine=object())
    assert len(writes) == 1 and len(writes[0]["settlements"]) == 2

    with pytest.raises(ValueError, match="Missing canonical NHL team name for NYR"):
        pipeline.fetch_results(day, teams.iloc[:1])


def test_daily_forecast_replays_offline_from_retained_inputs(tmp_path, monkeypatch):
    from datetime import timedelta

    from backend import backtest, replay
    from backend.etl import moneypuck, store
    from backend.model import goal_map
    from backend.model.ratings import InsufficientHistory
    from backend.odds.partner import OFFER_COLUMNS

    day = date(2026, 10, 5)
    games = pd.DataFrame(
        [
            dict(
                {c: 5.0 for c in moneypuck.COUNT_COLUMNS},
                team_abbr=team,
                venue=venue,
                season=2025,
                game_id=str(i),
                game_date=date(2026, 4, 1) + timedelta(days=i),
                situation=situation,
                toi=900.0,
            )
            for team in ("BOS", "NYR")
            for venue in ("home", "away")
            for i in range(30)
            for situation in ("ev", "pp", "pk", "other")
        ]
    )
    teams = pd.DataFrame(
        {"team_abbr": ["BOS", "NYR"], "team": ["Boston Bruins", "New York Rangers"]}
    )
    receipt = {"sha256": "a" * 64, "observed_at": "2026-10-05T00:00:00Z"}
    schedule = [
        dict(
            game_id="2026020001",
            season=2026,
            game_date=day,
            start_date="2026-10-06T02:00:00Z",
            home_abbr="BOS",
            away_abbr="NYR",
            home_team="Boston Bruins",
            away_team="New York Rangers",
        )
    ]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 5, 12, tzinfo=UTC).astimezone(tz)

    # The run stamps as_of from the wall clock and skips games already started.
    monkeypatch.setattr(pipeline, "datetime", Clock)
    monkeypatch.setattr(store, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(store, "RECEIPTS_DIR", tmp_path / "receipts")
    monkeypatch.setattr(pipeline, "ingest_moneypuck", lambda *a: receipt)
    monkeypatch.setattr(pipeline, "load_teams", lambda *a: (teams, receipt))
    monkeypatch.setattr(pipeline, "fetch_schedule", lambda *a: (schedule, receipt))
    monkeypatch.setattr(
        pipeline,
        "fetch_offers",
        lambda *a: (pd.DataFrame(columns=OFFER_COLUMNS), [receipt]),
    )
    monkeypatch.setattr(pipeline.verification, "verify", lambda offers, _: (offers, {}))
    monkeypatch.setattr(moneypuck, "read_games", lambda *a: games)
    summary = pipeline.daily(day)
    assert summary["projections"] == 1
    retained = store.read_receipt("forecast_replay")
    ratings, projections, decisions = replay.forecast(
        tmp_path / "receipts/sources" / (retained["sha256"] + ".gz")
    )
    assert projections[["home_fair_price", "away_fair_price"]].iloc[0].eq(100).all()
    for frame, kind in (
        (ratings, "ratings"),
        (projections, "projections"),
        (decisions, "decisions"),
    ):
        pd.testing.assert_frame_equal(
            frame,
            store.read_processed(kind, f"{day}.parquet"),
            check_dtype=False,
            atol=1e-12,
            rtol=1e-12,
        )
    # Expected history exclusions remain visible, while unexpected model errors fail.
    fixtures = pd.DataFrame(
        [
            dict(
                game_id="one",
                season=2026,
                game_date=day,
                home_abbr="BOS",
                away_abbr="NYR",
                home_goals=3,
                away_goals=2,
                last_period_type="REG",
            )
        ]
    )
    monkeypatch.setattr(backtest, "official_results", lambda *a: fixtures)
    monkeypatch.setattr(backtest.goal_map_model, "fit", lambda *a: goal_map.load())

    def short(*a):
        raise InsufficientHistory("no pregame history")

    monkeypatch.setattr(backtest, "team_ratings", short)
    result = backtest.run(games, [2026])
    assert result.empty and result.attrs["coverage"]["eligible"] == 1
    assert result.attrs["coverage"]["excluded"][0]["game_id"] == "one"

    def invalid(*a):
        raise ValueError("invalid source")

    monkeypatch.setattr(backtest, "team_ratings", invalid)
    with pytest.raises(ValueError, match="invalid source"):
        backtest.run(games, [2026])
