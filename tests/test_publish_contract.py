"""The publish column lists are the frontend contract; they must match the
checked-in DDL exactly, column for column."""

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from backend import pipeline, publish
from backend.backtest import BACKTEST_COLUMNS
from backend.etl import nhl_api
from backend.recommendations import RECOMMENDATION_COLUMNS, SETTLEMENT_COLUMNS, settle

TYPES = {"text", "int", "float8", "timestamptz", "date", "bool", "jsonb", "bigint"}
DDL = "\n".join(
    path.read_text()
    for path in sorted((Path(__file__).parent.parent / "sql").glob("*.sql"))
)

CONTRACTS = {
    "teams": publish.TEAMS_COLUMNS,
    "team_ratings": publish.TEAM_RATINGS_COLUMNS + ["published_at"],
    "game_projections": publish.GAME_PROJECTIONS_COLUMNS,
    "market_snapshots": publish.MARKET_SNAPSHOTS_COLUMNS,
    "game_results": publish.GAME_RESULTS_COLUMNS,
    "backtest_predictions": BACKTEST_COLUMNS,
    "recommendation_schedule": publish.RECOMMENDATION_SCHEDULE_COLUMNS,
}


def _ddl_columns(table: str) -> list[str]:
    match = re.search(rf"create table nhl\.{table} \((.*?)\n\);", DDL, re.DOTALL)
    assert match, f"table {table} not in DDL"
    columns = []
    for line in match.group(1).splitlines():
        tokens = line.split("--")[0].strip().rstrip(",").split()
        # Multi-line constraints never start with a column type in second place.
        if len(tokens) >= 2 and tokens[1] in TYPES:
            columns.append(tokens[0])
    # Additive migrations preserve the original table declaration.
    for alteration in re.finditer(
        rf"alter table nhl\.{table}\s+add column (\w+) ", DDL, re.IGNORECASE
    ):
        columns.append(alteration.group(1))
    return columns


def test_publish_columns_match_ddl():
    for table, contract in CONTRACTS.items():
        assert list(contract) == _ddl_columns(table), table


def test_recommendation_columns_match_ddl():
    # Decision columns come first; published_at is a database default and
    # the settlement columns follow it.
    ddl = _ddl_columns("recommendations")
    decision = [
        c for c in ddl if c not in SETTLEMENT_COLUMNS[2:] and c != "published_at"
    ]
    assert decision == RECOMMENDATION_COLUMNS
    assert ddl[-7:] == SETTLEMENT_COLUMNS[2:]


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
    with pytest.raises(ValueError, match="Missing canonical NHL team name for NYR"):
        pipeline.fetch_results(day, teams.iloc[:1])
