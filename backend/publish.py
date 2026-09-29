"""The only Supabase writer. Writes use per-key DELETE + append or upserts so
RLS, policies, and triggers survive; every statement is schema-qualified so a
lost search_path can never redirect a write. Timestamps become tz-aware
datetimes and every NaN becomes None: PostgREST cannot serialize NaN."""

import pandas as pd
from sqlalchemy import MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert

from backend.backtest import BACKTEST_COLUMNS
from backend.grading import RESULT_COLUMNS, SCHEDULE_COLUMNS
from backend.model.projections import PROJECTION_COLUMNS
from backend.odds.partner import SNAPSHOT_COLUMNS
from backend.recommendations import RECOMMENDATION_COLUMNS, SETTLEMENT_COLUMNS

SCHEMA = "nhl"

TEAMS_COLUMNS = [
    "team_abbr",
    "team",
    "conference",
    "division",
    "color",
    "logo_light",
    "logo_dark",
]
TEAM_RATINGS_COLUMNS = [
    "as_of",
    "team_abbr",
    "team",
    "model_version",
    "window_games_home",
    "window_games_away",
    "home_xgf",
    "home_xga",
    "away_xgf",
    "away_xga",
    "home_attack",
    "home_defense",
    "away_attack",
    "away_defense",
    "rating",
    "insufficient_window",
]
GAME_PROJECTIONS_COLUMNS = PROJECTION_COLUMNS
MARKET_SNAPSHOTS_COLUMNS = SNAPSHOT_COLUMNS
GAME_RESULTS_COLUMNS = RESULT_COLUMNS
RECOMMENDATION_SCHEDULE_COLUMNS = SCHEDULE_COLUMNS

_TIMESTAMP_COLUMNS = {
    "start_date",
    "as_of",
    "fetched_at",
    "provider_last_update",
    "source_fetched_at",
    "observed_at",
    "forecast_as_of",
    "decision_at",
    "market_fetched_at",
    "provider_start_date",
    "graded_at",
    "result_source_at",
}
_DATE_COLUMNS = {"as_of_date", "game_date"}


def _prepare(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.reindex(columns=columns)
    for column in columns:
        if column in _TIMESTAMP_COLUMNS:
            out[column] = pd.to_datetime(out[column], utc=True, format="ISO8601")
        elif column in _DATE_COLUMNS:
            out[column] = pd.to_datetime(out[column]).dt.date
        out[column] = out[column].astype(object).where(out[column].notna(), None)
    return out


def _append(frame: pd.DataFrame, table: str, conn) -> None:
    if frame.empty:
        return
    frame.to_sql(table, con=conn, schema=SCHEMA, if_exists="append", index=False)


def _upsert(frame: pd.DataFrame, table: str, conn, keys: list[str]) -> None:
    if frame.empty:
        return
    target = Table(table, MetaData(), schema=SCHEMA, autoload_with=conn)
    statement = insert(target).values(frame.to_dict("records"))
    conn.execute(
        statement.on_conflict_do_update(
            index_elements=keys,
            set_={c: statement.excluded[c] for c in frame.columns if c not in keys},
        )
    )


def _insert_ignore(frame: pd.DataFrame, table: str, conn, keys: list[str]) -> None:
    if frame.empty:
        return
    target = Table(table, MetaData(), schema=SCHEMA, autoload_with=conn)
    statement = insert(target).values(frame.to_dict("records"))
    conn.execute(statement.on_conflict_do_nothing(index_elements=keys))


def pending_decisions(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text(
                f"SELECT game_id, market, status, side, point, price, stake_units, "
                f"start_date FROM {SCHEMA}.recommendations WHERE outcome = 'pending'"
            ),
            conn,
        )


def publish_day(
    engine,
    as_of,
    teams: pd.DataFrame | None = None,
    ratings: pd.DataFrame | None = None,
    projections: pd.DataFrame | None = None,
    market_snapshot: pd.DataFrame | None = None,
    results: pd.DataFrame | None = None,
    schedule_observation: pd.DataFrame | None = None,
    recommendations: pd.DataFrame | None = None,
    settlements: pd.DataFrame | None = None,
) -> dict[str, int]:
    """One transaction. Projections and results upsert by game; the database
    archives pregame revisions and rejects late ones. New decisions insert
    once and pending No Play rows may be replaced before puck drop."""
    counts: dict[str, int] = {}
    with engine.begin() as conn:
        if teams is not None:
            conn.execute(text(f"DELETE FROM {SCHEMA}.teams"))
            _append(_prepare(teams, TEAMS_COLUMNS), "teams", conn)
            counts["teams"] = len(teams)
        if ratings is not None:
            conn.execute(
                text(f"DELETE FROM {SCHEMA}.team_ratings WHERE as_of = :d"),
                {"d": pd.Timestamp(as_of).date()},
            )
            _append(_prepare(ratings, TEAM_RATINGS_COLUMNS), "team_ratings", conn)
            counts["team_ratings"] = len(ratings)
        if projections is not None:
            _upsert(
                _prepare(projections, GAME_PROJECTIONS_COLUMNS),
                "game_projections",
                conn,
                ["game_id"],
            )
            counts["game_projections"] = len(projections)
        if market_snapshot is not None:
            _insert_ignore(
                _prepare(market_snapshot, MARKET_SNAPSHOTS_COLUMNS),
                "market_snapshots",
                conn,
                ["game_id", "provider_key", "fetched_at"],
            )
            counts["market_snapshots"] = len(market_snapshot)
        if results is not None:
            _upsert(
                _prepare(results, GAME_RESULTS_COLUMNS),
                "game_results",
                conn,
                ["game_id"],
            )
            counts["game_results"] = len(results)
        if schedule_observation is not None:
            _upsert(
                _prepare(schedule_observation, RECOMMENDATION_SCHEDULE_COLUMNS),
                "recommendation_schedule",
                conn,
                ["game_id"],
            )
            counts["recommendation_schedule"] = len(schedule_observation)
        if recommendations is not None:
            frame = _prepare(recommendations, RECOMMENDATION_COLUMNS)
            existing = pd.read_sql(
                text(
                    f"SELECT game_id, market, status, outcome "
                    f"FROM {SCHEMA}.recommendations"
                ),
                conn,
            )
            keyed = existing.set_index(["game_id", "market"])
            new_rows, replace_rows = [], []
            for row in frame.to_dict("records"):
                key = (row["game_id"], row["market"])
                if key not in keyed.index:
                    new_rows.append(row)
                elif (
                    keyed.loc[key, "status"] == "no_play"
                    and keyed.loc[key, "outcome"] == "pending"
                ):
                    replace_rows.append(row)
            target = Table(
                "recommendations", MetaData(), schema=SCHEMA, autoload_with=conn
            )
            if new_rows:
                conn.execute(insert(target).values(new_rows))
            for row in replace_rows:
                conn.execute(
                    target.update()
                    .where(target.c.game_id == row["game_id"])
                    .where(target.c.market == row["market"])
                    .where(target.c.status == "no_play")
                    .where(target.c.outcome == "pending")
                    .values(
                        {k: v for k, v in row.items() if k not in ("game_id", "market")}
                    )
                )
            counts["recommendations_new"] = len(new_rows)
            counts["recommendations_replaced"] = len(replace_rows)
        if settlements is not None:
            frame = _prepare(settlements, SETTLEMENT_COLUMNS)
            target = Table(
                "recommendations", MetaData(), schema=SCHEMA, autoload_with=conn
            )
            for row in frame.to_dict("records"):
                conn.execute(
                    target.update()
                    .where(target.c.game_id == row["game_id"])
                    .where(target.c.market == row["market"])
                    .where(target.c.outcome == "pending")
                    .values(
                        {k: v for k, v in row.items() if k not in ("game_id", "market")}
                    )
                )
            counts["settlements"] = len(frame)
    return counts


def publish_backtest(engine, frame: pd.DataFrame) -> int:
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {SCHEMA}.backtest_predictions"))
        _append(_prepare(frame, BACKTEST_COLUMNS), "backtest_predictions", conn)
    return len(frame)
