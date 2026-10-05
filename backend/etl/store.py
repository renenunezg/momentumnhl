"""Parquet store under backend/data plus source receipts every stage shares."""

import gzip
import hashlib
import json
import os
from datetime import UTC, datetime

import pandas as pd

from backend.config import PROCESSED_DIR, RAW_DIR

RECEIPTS_DIR = PROCESSED_DIR / "receipts"


def write_parquet(df: pd.DataFrame, path) -> None:
    """Atomic parquet write: tmp file then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def raw_path(*parts: str):
    return RAW_DIR.joinpath(*parts)


def write_raw(df: pd.DataFrame, *parts: str) -> None:
    write_parquet(df, RAW_DIR.joinpath(*parts))


def read_raw(*parts: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR.joinpath(*parts), columns=columns)


def write_processed(df: pd.DataFrame, *parts: str) -> None:
    write_parquet(df, PROCESSED_DIR.joinpath(*parts))


def read_processed(*parts: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR.joinpath(*parts), columns=columns)


def receipt(
    name: str,
    content: bytes,
    observed_at: datetime | None = None,
    *,
    archive: bool = True,
) -> dict:
    """Identify each fetch and retain compact source bodies for replay."""
    observed = (observed_at or datetime.now(UTC)).astimezone(UTC)
    record = {
        "name": name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "observed_at": observed.isoformat().replace("+00:00", "Z"),
        "bytes": len(content),
    }
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    if archive:
        target = RECEIPTS_DIR / "sources" / (record["sha256"] + ".gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(gzip.compress(content, mtime=0))
    record["body_archived"] = archive
    history = RECEIPTS_DIR / "history"
    history.mkdir(exist_ok=True)
    identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    (history / f"{identity}.json").write_text(json.dumps(record, indent=1))
    (RECEIPTS_DIR / f"{name}.json").write_text(json.dumps(record, indent=1))
    return record


def read_receipt(name: str) -> dict | None:
    path = RECEIPTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
