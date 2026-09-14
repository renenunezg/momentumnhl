"""The sheet's raw data block: each team's last 25 games at a venue, summed by
situation, then turned into the per-60 rates and percentages the goal map
reads. Windows are built strictly before an as-of date and run back into the
previous season so October is never empty."""

import numpy as np
import pandas as pd

from backend.config import MIN_WINDOW_GAMES, WINDOW_GAMES
from backend.etl.moneypuck import COUNT_COLUMNS

MODEL_SITUATIONS = ("ev", "pp", "pk")
FOR_FEATURES = [
    "sf60", "scsh_pct", "hdgf60", "hdsh_pct", "mdsf60", "mdsh_pct",
    "mdgf60", "ldsf60", "ldsh_pct", "sh_pct",
]  # fmt: skip
AGAINST_FEATURES = [
    "sa60", "scsv_pct", "hdsa60", "hdga60", "mdsa60", "mdga60",
    "ldsa60", "ldga60", "ldsv_pct", "sv_pct",
]  # fmt: skip


def venue_windows(
    games: pd.DataFrame,
    as_of,
    window: int = WINDOW_GAMES,
    minimum: int = MIN_WINDOW_GAMES,
) -> pd.DataFrame:
    """One row per (team_abbr, venue, situation) with summed counts over the
    last `window` games at that venue before `as_of`, the games counted,
    ice time per game, and whether the window is too short to price."""
    rows = games[
        games["situation"].isin(MODEL_SITUATIONS) & (games["game_date"] < as_of)
    ]
    # Rank each team-venue's games newest first on the EV rows, then keep
    # every situation row of the selected games.
    keys = rows.loc[
        rows["situation"].eq("ev"), ["team_abbr", "venue", "game_date", "game_id"]
    ].sort_values(["team_abbr", "venue", "game_date", "game_id"], ascending=False)
    keys["rank"] = keys.groupby(["team_abbr", "venue"]).cumcount()
    keep = keys.loc[keys["rank"] < window, ["team_abbr", "venue", "game_id"]]
    selected = rows.merge(keep, on=["team_abbr", "venue", "game_id"])
    summed = selected.groupby(["team_abbr", "venue", "situation"], as_index=False)[
        COUNT_COLUMNS
    ].sum()
    counts = keep.groupby(["team_abbr", "venue"]).size().rename("games").reset_index()
    summed = summed.merge(counts, on=["team_abbr", "venue"])
    summed["toi_per_game"] = summed["toi"] / summed["games"]
    summed["insufficient"] = summed["games"] < minimum
    return summed


def _per60(count: pd.Series, toi: pd.Series) -> pd.Series:
    return np.where(toi > 0, count * 3600 / toi.replace(0, np.nan), 0.0)


def _pct(hits: pd.Series, attempts: pd.Series) -> pd.Series:
    return hits / attempts.replace(0, np.nan)


def feature_rates(windows: pd.DataFrame) -> pd.DataFrame:
    """Adds the ten 'for' and ten 'against' features plus the goals-per-60
    targets. A percentage with no attempts is filled with the league mean of
    that column so a quiet penalty kill does not read as perfect."""
    out = windows.copy()
    toi = out["toi"]
    out["sf60"] = _per60(out["sf"], toi)
    out["hdgf60"] = _per60(out["hd_gf"], toi)
    out["mdsf60"] = _per60(out["md_sf"], toi)
    out["mdgf60"] = _per60(out["md_gf"], toi)
    out["ldsf60"] = _per60(out["ld_sf"], toi)
    out["sh_pct"] = _pct(out["gf"], out["sf"])
    out["hdsh_pct"] = _pct(out["hd_gf"], out["hd_sf"])
    out["mdsh_pct"] = _pct(out["md_gf"], out["md_sf"])
    out["ldsh_pct"] = _pct(out["ld_gf"], out["ld_sf"])
    out["scsh_pct"] = _pct(out["hd_gf"] + out["md_gf"], out["hd_sf"] + out["md_sf"])
    out["sa60"] = _per60(out["sa"], toi)
    out["hdsa60"] = _per60(out["hd_sa"], toi)
    out["hdga60"] = _per60(out["hd_ga"], toi)
    out["mdsa60"] = _per60(out["md_sa"], toi)
    out["mdga60"] = _per60(out["md_ga"], toi)
    out["ldsa60"] = _per60(out["ld_sa"], toi)
    out["ldga60"] = _per60(out["ld_ga"], toi)
    out["sv_pct"] = 1 - _pct(out["ga"], out["sa"])
    out["ldsv_pct"] = 1 - _pct(out["ld_ga"], out["ld_sa"])
    out["scsv_pct"] = 1 - _pct(out["hd_ga"] + out["md_ga"], out["hd_sa"] + out["md_sa"])
    out["gf60"] = _per60(out["gf"], toi)
    out["ga60"] = _per60(out["ga"], toi)
    pct_columns = [c for c in FOR_FEATURES + AGAINST_FEATURES if c.endswith("_pct")]
    for column in pct_columns:
        means = out.groupby("situation")[column].transform("mean")
        out[column] = out[column].fillna(means).fillna(0.0)
    return out
