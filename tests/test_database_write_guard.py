"""Production connection permissions, including the first and reused transaction."""

import os

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError

from backend import db


def test_connection_enforces_write_authorization(monkeypatch):
    value = os.getenv("NHL_TEST_DATABASE_URL")
    if not value:
        pytest.skip("requires disposable local PostgreSQL")
    assert make_url(value).host in ("127.0.0.1", "localhost")
    monkeypatch.setenv("DATABASE_URL", value)
    for human, ci, allowed in (
        ("0", "false", False),
        ("1", "false", True),
        ("0", "true", True),
    ):
        monkeypatch.setenv("MOMENTUMNHL_DB_WRITES", human)
        monkeypatch.setenv("GITHUB_ACTIONS", ci)
        monkeypatch.setattr(db, "_engine", None)
        engine = db.engine
        try:
            # A comment bypasses the fast SQL-prefix check; PostgreSQL must
            # enforce permissions before the first transaction and after reuse.
            for _ in range(2):
                with engine.connect() as conn:
                    transaction = conn.begin()
                    try:
                        assert conn.execute(text("SELECT 1")).scalar_one() == 1
                        statement = text(
                            "/* acceptance */ CREATE TABLE "
                            "public.nhl_write_guard_acceptance (id int)"
                        )
                        if allowed:
                            conn.execute(statement)
                        else:
                            with pytest.raises(DBAPIError, match="read-only"):
                                conn.execute(statement)
                    finally:
                        transaction.rollback()
        finally:
            engine.dispose()
