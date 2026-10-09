"""An independently corroborated quote must survive both decision and DB gates."""

import gzip
import hashlib
import json
import os
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import requests
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

from backend import publish
from backend.model.projections import PROJECTION_COLUMNS
from backend.odds import partner, verification
from backend.recommendations import RECOMMENDATION_COLUMNS, decide


@pytest.fixture
def quote_case(monkeypatch, tmp_path):
    now = datetime.now(UTC).replace(microsecond=0) - timedelta(seconds=1)
    start = (now + timedelta(hours=4)).replace(second=0)
    local = (start + timedelta(minutes=10)).astimezone(ZoneInfo("America/New_York"))
    html = (Path(__file__).parent / "fixtures/draftkings_listing.html").read_text()
    html = html.replace("9/29, 05:10PM", f"{local.month}/{local.day}, {local:%I:%M%p}")
    game = dict(
        game_id="2026020001",
        start_date=start.isoformat(),
        home_team="Carolina Hurricanes",
        away_team="Florida Panthers",
        home_abbr="CAR",
        away_abbr="FLA",
    )
    base = dict(
        game_id=game["game_id"],
        provider="DraftKings",
        provider_key="draftkings",
        provider_last_update=(now - timedelta(days=2)).isoformat(),
        fetched_at=(now - timedelta(seconds=3)).isoformat(),
        provider_start_date=start.isoformat(),
    )
    offers = pd.DataFrame(
        [
            dict(base, market=m, side=s, point=p, price=v)
            for m, s, p, v in (
                ("h2h", "home", None, -130),
                ("h2h", "away", None, 110),
                ("totals", "over", 6.5, 105),
                ("totals", "under", 6.5, -125),
            )
        ]
    )
    row = dict.fromkeys(PROJECTION_COLUMNS)
    row.update(
        game_id=game["game_id"],
        season=2026,
        game_date=start.date(),
        start_date=start.isoformat(),
        as_of=(now - timedelta(seconds=4)).isoformat(),
        model_version="nhl-poisson-v1.1",
        home_team_abbr="CAR",
        away_team_abbr="FLA",
        home_team=game["home_team"],
        away_team=game["away_team"],
        home_lambda=4.5,
        away_lambda=3.4,
        home_win_prob=0.85,
        away_win_prob=0.15,
        model_total=7.9,
        missing_input_count=0,
        pure_home_win_prob=0.85,
        pure_model_total=7.9,
        market_home_prob=0.54,
        market_total=6.5,
    )
    projection = pd.DataFrame([row])
    receipts = {
        s: dict(sha256="a" * 64, observed_at=base["fetched_at"])
        for s in ("moneypuck", "schedule", "odds")
    }

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    class Response:
        status_code = 200
        headers = {"Date": format_datetime(now), "Content-Type": "text/html"}
        body = html.encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, size):
            yield self.body

    response = Response()
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        assert url == verification.SOURCE_URL and kwargs["allow_redirects"] is False
        return response

    monkeypatch.setattr(verification, "datetime", Clock)
    monkeypatch.setattr(verification.requests, "get", get)
    monkeypatch.setattr(verification, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(verification.store, "RECEIPTS_DIR", tmp_path / "receipts")
    return SimpleNamespace(**locals())


def test_live_listing_to_decision_fails_closed(quote_case):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/contracts/v1/nhl-quote.json").read_text()
    )
    assert fixture["version"] == 1
    projection = SimpleNamespace(**fixture["forecast"])
    snapshot = fixture["snapshot"]
    for evidence in snapshot["quote_verifications"].values():
        offer = SimpleNamespace(
            **{
                **evidence,
                "fetched_at": snapshot["fetched_at"],
                "provider_start_date": projection.start_date,
            }
        )
        assert verification.valid(evidence, offer, projection, projection.as_of)
        for mutation in fixture["rejected_evidence"]:
            assert not verification.valid(
                {**evidence, **mutation}, offer, projection, projection.as_of
            )
    c = quote_case
    checked, report = verification.verify(c.offers, [c.game])
    assert report["verified"] == 4 and len(c.calls) == 1
    snapshot = partner.market_snapshot(checked).iloc[0]
    mismatched = checked.copy()
    mismatched.loc[mismatched.side.eq("under"), "point"] += 0.5
    invalid = partner.market_snapshot(mismatched).iloc[0]
    assert pd.isna(invalid.total_line)
    assert pd.isna(invalid.over_price) and pd.isna(invalid.under_price)
    assert not any(k.startswith("totals_") for k in invalid.quote_verifications)
    assert set(snapshot.quote_verifications) == {
        "h2h_home",
        "h2h_away",
        "totals_over",
        "totals_under",
    }
    assert snapshot.quote_verifications["h2h_away"]["price"] == 110
    assert partner.market_snapshot(c.offers).iloc[0].quote_verifications == {}
    archived = c.tmp_path / "raw/quote_verification" / (report["sha256"] + ".gz")
    assert (
        hashlib.sha256(gzip.decompress(archived.read_bytes())).hexdigest()
        == report["sha256"]
    )
    decisions = decide(c.projection, checked, c.receipts, c.now.isoformat())
    assert decisions.status.eq("recommended").all()
    assert decisions.provider_last_update.eq(c.base["provider_last_update"]).all()
    assert (
        decisions.data_flags.map(lambda f: f["quote_verification"]["sha256"])
        .eq(report["sha256"])
        .all()
    )
    assert (
        decide(c.projection, c.offers, c.receipts, c.now.isoformat())
        .reason.eq("stale_offer")
        .all()
    )
    assert (
        decide(
            c.projection,
            checked,
            c.receipts,
            (c.now + timedelta(minutes=16)).isoformat(),
        )
        .reason.eq("stale_offer")
        .all()
    )

    # Every field that could accidentally price the wrong bet is bound.
    for column, value in (
        ("price", 120),
        ("point", 5.5),
        ("side", "away"),
        ("provider_key", "fanduel"),
        ("game_id", "wrong"),
        ("start_date", (c.start + timedelta(days=1)).isoformat()),
        ("source_start_date", (c.start + timedelta(hours=1)).isoformat()),
        ("observed_at", (c.now + timedelta(seconds=1)).isoformat()),
    ):
        bad = deepcopy(checked)
        bad["quote_verification"] = checked.quote_verification.map(
            lambda e: dict(e, **{column: value})
        )
        got = decide(c.projection, bad, c.receipts, c.now.isoformat())
        assert got.status.eq("no_play").all(), column

    receipt = {
        k: report[k]
        for k in (
            "source_url",
            "sha256",
            "observed_at",
            "response_date",
            "response_age_seconds",
        )
    }
    for html in (
        c.html.replace('value="42133"', 'value="88808"'),
        c.html.replace(
            "FLA Panthers @ CAR Hurricanes", "CAR Hurricanes @ FLA Panthers"
        ),
        c.html.replace("Moneyline", "1st Period Moneyline").replace(
            "Total", "1st Period Total"
        ),
        c.html.replace("0ML", "1ML").replace("0OU", "1OU"),
        c.html.replace("outcomes=", "disabled="),
        c.html + c.html,
        c.html.replace("6.5", "9" * 400).replace("Moneyline", "Unsupported"),
    ):
        assert (
            verification.corroborate(html, receipt, c.offers, [c.game])
            .quote_verification.isna()
            .all()
        )

    for status, body, headers in (
        (403, b"Access denied", c.response.headers),
        (200, b"<html>layout changed</html>", c.response.headers),
        (
            200,
            c.html.encode(),
            {**c.response.headers, "Date": format_datetime(c.now - timedelta(hours=1))},
        ),
        (200, c.html.encode(), {**c.response.headers, "Age": "301"}),
        (
            200,
            c.html.encode(),
            {**c.response.headers, "Content-Type": "application/json"},
        ),
    ):
        c.response.status_code, c.response.body, c.response.headers = (
            status,
            body,
            headers,
        )
        rejected, result = verification.verify(c.offers, [c.game])
        assert result["verified"] == 0
        assert (
            decide(c.projection, rejected, c.receipts, c.now.isoformat())
            .status.eq("no_play")
            .all()
        )
        assert (
            json.loads(
                (c.tmp_path / "receipts/quote_verification_status.json").read_text()
            )
            == result
        )

    def timeout(*args, **kwargs):
        raise requests.Timeout()

    c.monkeypatch.setattr(verification.requests, "get", timeout)
    _, result = verification.verify(c.offers, [c.game])
    assert result["status"] == "source_unavailable"
    assert (
        json.loads((c.tmp_path / "receipts/quote_verification_status.json").read_text())
        == result
    )


def test_database_enforces_publication_evidence(quote_case):
    url = os.environ.get("NHL_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires disposable local PostgreSQL with sql/*.sql applied")
    assert make_url(url).host in ("localhost", "127.0.0.1")
    c = quote_case
    checked, _ = verification.verify(c.offers, [c.game])
    decisions = decide(c.projection, checked, c.receipts, c.now.isoformat())
    records = publish._prepare(decisions, RECOMMENDATION_COLUMNS).to_dict("records")
    engine = create_engine(url)
    with engine.connect() as conn:
        transaction = conn.begin()
        from sqlalchemy import inspect

        from backend.backtest import BACKTEST_COLUMNS
        from backend.recommendations import SETTLEMENT_COLUMNS

        contracts = {
            "recommendations": RECOMMENDATION_COLUMNS + SETTLEMENT_COLUMNS[2:],
            "teams": publish.TEAMS_COLUMNS,
            "team_ratings": publish.TEAM_RATINGS_COLUMNS + ["published_at"],
            "game_projections": publish.GAME_PROJECTIONS_COLUMNS,
            "market_snapshots": publish.MARKET_SNAPSHOTS_COLUMNS,
            "game_results": publish.GAME_RESULTS_COLUMNS,
            "backtest_predictions": BACKTEST_COLUMNS,
            "recommendation_schedule": publish.RECOMMENDATION_SCHEDULE_COLUMNS,
        }
        inspector = inspect(conn)
        for table, columns in contracts.items():
            stored = {
                column["name"] for column in inspector.get_columns(table, schema="nhl")
            }
            assert set(columns) <= stored, table
        target = Table("recommendations", MetaData(), schema="nhl", autoload_with=conn)
        snapshots = partner.market_snapshot(checked)
        publish._insert_ignore(
            publish._prepare(snapshots, publish.MARKET_SNAPSHOTS_COLUMNS),
            "market_snapshots",
            conn,
            ["game_id", "provider_key", "fetched_at"],
        )
        archived = conn.execute(
            text("select quote_verifications from nhl.market_snapshots")
        ).scalar_one()
        assert len(archived) == 4 and archived["h2h_away"]["price"] == 110
        conn.execute(
            text("""INSERT INTO nhl.market_snapshots
            (game_id, provider_key, fetched_at, home_price)
            SELECT :game, 'draftkings',
              CAST(:cutoff AS timestamptz) + n * interval '1 second', 999
            FROM generate_series(1, 10001) n"""),
            {"game": c.game["game_id"], "cutoff": c.now},
        )
        latest = (
            conn.execute(
                text("""SELECT * FROM nhl.latest_market_snapshots(
            ARRAY[:game], jsonb_build_object(:game, CAST(:cutoff AS text)))"""),
                {"game": c.game["game_id"], "cutoff": c.now},
            )
            .mappings()
            .all()
        )
        assert len(latest) == 1 and latest[0]["home_price"] == -130
        assert latest[0]["quote_verifications"] == archived
        for row in records:
            for field, value in (
                ("sha256", "bad"),
                ("price", 500),
                ("game_id", "wrong"),
                ("point", 9.5),
                ("method", "unknown"),
                ("observed_at", (c.now - timedelta(minutes=16)).isoformat()),
                ("observed_at", "not-a-date"),
                ("source_selection_id", "1ML123_1"),
            ):
                bad = deepcopy(row)
                bad["data_flags"]["quote_verification"][field] = value
                with pytest.raises(IntegrityError), conn.begin_nested():
                    conn.execute(target.insert().values(**bad))
            conn.execute(target.insert().values(**row))
        saved = conn.execute(
            text("select data_flags, provider_last_update from nhl.recommendations")
        ).all()
        assert len(saved) == 2
        assert all(
            r.data_flags["quote_verification"]["method"] == verification.METHOD
            for r in saved
        )
        with pytest.raises(Exception, match="frozen"), conn.begin_nested():
            conn.execute(
                text("update nhl.recommendations set data_flags = '{}'::jsonb")
            )
        # Real provider observation -> normalized PPD -> SQL-backed void.
        from backend import grading
        from backend.recommendations import settle

        observed = datetime.now(UTC)
        score_rows = [
            {
                "game_id": c.game["game_id"],
                "season": 2026,
                "start_date": c.start.isoformat(),
                "home_team": c.game["home_team"],
                "away_team": c.game["away_team"],
                "schedule_state": "PPD",
                "game_state": "FUT",
            }
        ]
        empty = pd.DataFrame(columns=["game_id"])
        observation = grading.schedule_observation(score_rows, empty, observed)
        publish._upsert(
            publish._prepare(observation, publish.RECOMMENDATION_SCHEDULE_COLUMNS),
            "recommendation_schedule",
            conn,
            ["game_id"],
        )
        settlements = publish._prepare(
            settle(decisions, empty, observation, datetime.now(UTC)),
            publish.SETTLEMENT_COLUMNS,
        )
        for row in settlements.to_dict("records"):
            conn.execute(
                target.update()
                .where(target.c.game_id == row["game_id"])
                .where(target.c.market == row["market"])
                .values(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in ("game_id", "market")
                    }
                )
            )
        assert conn.execute(
            text("select outcome from nhl.recommendations")
        ).scalars().all() == ["void", "void"]
        transaction.rollback()
    engine.dispose()
