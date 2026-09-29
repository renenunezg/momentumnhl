"""The production write gate blocks writes unless CI or a human opts in."""

from __future__ import annotations

import os

from sqlalchemy import text

from backend import db
from backend.db import _is_write_statement, writes_allowed


def test_writes_allowed_logic(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("MOMENTUMNHL_DB_WRITES", raising=False)
    assert writes_allowed() is False
    monkeypatch.setenv("MOMENTUMNHL_DB_WRITES", "1")
    assert writes_allowed() is True
    monkeypatch.delenv("MOMENTUMNHL_DB_WRITES", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert writes_allowed() is True
    if os.getenv("DATABASE_URL"):
        monkeypatch.setenv("GITHUB_ACTIONS", "false")
        monkeypatch.setattr(db, "_engine", None)
        engine = db.engine
        try:
            # The first transaction and a reused connection must both be
            # read-only; SET inside a transaction missed the first one.
            for _ in range(2):
                with engine.connect() as conn:
                    read_only = conn.execute(
                        text("show transaction_read_only")
                    ).scalar()
                    assert read_only == "on"
        finally:
            engine.dispose()


def test_is_write_statement():
    for w in (
        "INSERT INTO x VALUES (1)",
        "  \n  UPDATE x SET a=1",
        "DELETE FROM x",
        "TRUNCATE TABLE x",
        "drop table x",
        "ALTER TABLE x ADD c INT",
    ):
        assert _is_write_statement(w) is True, w
    for r in (
        "SELECT * FROM x",
        "  WITH c AS (SELECT 1) SELECT * FROM c",
        "BEGIN",
        "COMMIT",
    ):
        assert _is_write_statement(r) is False, r
