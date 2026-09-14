"""Rolling venue windows: newest 25 games, previous-season spillover, and the
minimum-window flag."""

from datetime import date, timedelta

import pandas as pd
import pytest

from backend.etl.moneypuck import COUNT_COLUMNS
from backend.features.windows import feature_rates, venue_windows


def _games(team: str, venue: str, count: int, first: date, season_split: int):
    rows = []
    for index in range(count):
        day = first + timedelta(days=3 * index)
        season = 2025 if index < season_split else 2026
        for situation, toi in (("ev", 2900.0), ("pp", 300.0), ("pk", 250.0)):
            row = {c: 0.0 for c in COUNT_COLUMNS}
            row.update(
                team_abbr=team,
                opponent="XXX",
                venue=venue,
                season=season,
                game_id=f"{season}02{index:04d}",
                game_date=day,
                situation=situation,
                toi=toi,
                sf=float(index + 1),
                gf=1.0,
                sa=20.0,
                ga=2.0,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_window_keeps_newest_25_home_games_across_seasons():
    frame = _games("TOR", "home", 30, date(2025, 10, 1), season_split=20)
    as_of = frame["game_date"].max()  # the newest game itself is excluded
    windows = venue_windows(frame, as_of)
    ev = windows[windows["situation"].eq("ev")].iloc[0]
    assert ev["games"] == 25
    assert not ev["insufficient"]
    # Games 4..28 (zero-based) are the 25 newest before the last one.
    assert ev["sf"] == sum(range(5, 30))
    assert ev["toi_per_game"] == pytest.approx(2900.0)
    rates = feature_rates(windows)
    ev_rates = rates[rates["situation"].eq("ev")].iloc[0]
    assert ev_rates["sf60"] == pytest.approx(sum(range(5, 30)) * 3600 / (25 * 2900))
    assert ev_rates["sv_pct"] == pytest.approx(0.9)


def test_short_window_is_flagged():
    frame = _games("BOS", "away", 8, date(2026, 10, 1), season_split=0)
    windows = venue_windows(frame, date(2026, 12, 1))
    assert windows["games"].eq(8).all()
    assert windows["insufficient"].all()
