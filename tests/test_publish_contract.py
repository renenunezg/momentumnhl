"""The publish column lists are the frontend contract; they must match the
checked-in DDL exactly, column for column."""

import re
from pathlib import Path

from backend import publish
from backend.backtest import BACKTEST_COLUMNS
from backend.recommendations import RECOMMENDATION_COLUMNS, SETTLEMENT_COLUMNS

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
