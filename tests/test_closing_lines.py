"""The close is the latest valid pregame quote, written once as the game starts."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd

from backend.odds import partner
from backend.odds.closing import ClosingArchiver, closing_quotes, newer

START = datetime(2026, 10, 4, 23, 0, tzinfo=UTC)


@dataclass(frozen=True)
class Fixture:
    game_id: str
    season: int
    start: datetime


def _feed(stamp: datetime, fetched: datetime, games: dict[str, int], **prices) -> dict:
    """One FanDuel feed; `games` maps game ids to the home moneyline, and a
    None price leaves that game one-sided."""
    return {
        "provider": "FanDuel",
        "provider_key": prices.get("provider_key", "fanduel"),
        "last_update": stamp.isoformat(),
        "fetched_at": fetched.isoformat(),
        "games": [
            {
                "game_id": game_id,
                "start_date": START.isoformat(),
                "home": [
                    {"description": "MONEY_LINE_2_WAY", "value": home},
                    {"description": "OVER_UNDER", "qualifier": "O6.5", "value": -110},
                    {"description": "PUCK_LINE", "qualifier": "-1.5", "value": 180},
                ],
                "away": [
                    {"description": "MONEY_LINE_2_WAY", "value": 110},
                    {"description": "OVER_UNDER", "qualifier": "U6.5", "value": -110},
                    {"description": "PUCK_LINE", "qualifier": "1.5", "value": -220},
                ],
            }
            for game_id, home in games.items()
        ],
    }


def _offers(*feeds: dict) -> pd.DataFrame:
    return pd.concat([partner.offers(feed) for feed in feeds], ignore_index=True)


def test_only_pregame_quotes_inside_the_window_qualify():
    minutes = timedelta(minutes=1)
    game = Fixture("1", 2026, START)

    def quotes(stamp, fetched, home=-130):
        return closing_quotes(_offers(_feed(stamp, fetched, {"1": home})), [game])

    assert quotes(START - 16 * minutes, START - 15 * minutes) == []
    assert quotes(START, START - 1 * minutes) == []
    assert quotes(START - 1 * minutes, START) == []
    assert quotes(START - 1 * minutes, START - 2 * minutes) == []
    assert quotes(START - 1 * minutes, START - 1 * minutes, home=None) == []
    [close] = quotes(START - 15 * minutes, START - 14 * minutes)
    assert close["source"] == "nhl_partner:fanduel"
    assert close["observed_at"] == START - 15 * minutes
    assert (close["home_moneyline"], close["away_moneyline"]) == (-130, 110)
    assert (close["total"], close["over_price"]) == (6.5, -110)
    assert (close["home_spread"], close["spread_home_price"]) == (-1.5, 180)
    assert close["spread_away_price"] == -220

    earlier = {"observed_at": START - 10 * minutes, "fetched_at": START - 9 * minutes}
    later = {"observed_at": START - 5 * minutes, "fetched_at": START - 9 * minutes}
    refetched = {"observed_at": START - 10 * minutes, "fetched_at": START - 8 * minutes}
    assert newer(later, earlier) and not newer(earlier, later)
    assert newer(refetched, earlier) and not newer(earlier, refetched)
    assert newer(earlier, None) and not newer(earlier, dict(earlier))


def test_each_game_writes_its_latest_quote_once_when_it_starts():
    minutes = timedelta(minutes=1)
    games = [Fixture("1", 2026, START), Fixture("2", 2026, START)]
    feed = {"stamp": START - 14 * minutes, "home": {"1": -130, "2": None}}
    fetches, written = [], []

    def fetch_offers():
        fetches.append(True)
        return _offers(_feed(feed["stamp"], now, feed["home"])), []

    archiver = ClosingArchiver(fetch_offers, written.append)

    # Outside the window the feed is left alone.
    now = START - 16 * minutes
    archiver.capture(games, now)
    assert fetches == [] and archiver.pending == {}

    now = START - 14 * minutes
    archiver.capture(games, now)
    assert len(fetches) == 1 and set(archiver.pending) == {"1"}

    # A newer stamp replaces the held quote; an older restamp does not.
    feed["stamp"], feed["home"] = START - 9 * minutes, {"1": -140, "2": -105}
    now = START - 9 * minutes
    archiver.capture(games, now)
    assert archiver.pending["1"]["home_moneyline"] == -140
    assert archiver.pending["2"]["home_moneyline"] == -105
    feed["stamp"], feed["home"] = START - 10 * minutes, {"1": -150, "2": None}
    now = START - 8 * minutes
    archiver.capture(games, now)
    assert archiver.pending["1"]["home_moneyline"] == -140

    # The in-play stamp at puck drop never qualifies.
    feed["stamp"], feed["home"] = START, {"1": -500, "2": -500}
    now = START - 30 * timedelta(seconds=1)
    archiver.capture(games, now)
    assert archiver.pending["1"]["home_moneyline"] == -140 and written == []

    # Nothing is written before the start; the first poll after it writes
    # one row per started game, and later polls write nothing more.
    for now in (START + 1 * minutes, START + 2 * minutes, START + 3 * minutes):
        archiver.capture(games, now)
    [closes] = written
    assert list(closes["game_id"]) == ["1", "2"]
    assert closes.set_index("game_id").loc["1", "home_moneyline"] == -140
    assert closes.loc[0, "observed_at"] == START - 9 * minutes
    assert closes.loc[0, "start_date"] == START and closes.loc[0, "season"] == 2026
    assert archiver.pending == {}

    # A game dropped from the slate releases its quote.
    now = START - 5 * minutes
    archiver.next_fetch = None
    feed["stamp"], feed["home"] = START - 5 * minutes, {"1": -130, "2": -130}
    archiver.capture(games, now)
    assert set(archiver.pending) == {"1", "2"}
    archiver.capture(games[:1], now)
    assert set(archiver.pending) == {"1"}
