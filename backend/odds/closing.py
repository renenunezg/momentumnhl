"""Captures each game's closing line from the partner feed. While a game is
inside the closing window the worker keeps that game's latest valid pregame
quote in memory; at the first poll after the scheduled start it writes that one
quote to nhl.closing_lines, which the database then freezes. The feed turns to
in-play lines once a game starts, so nothing stamped or fetched at or after
the start is ever a close, and a game with no valid quote gets no row."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from backend.config import CLOSING_LINE_MINUTES
from backend.odds import partner

# FanDuel restamps its feed every five minutes and DraftKings hourly.
POLL_SECONDS = 60
SOURCE_PREFIX = "nhl_partner:"

# The sql/ contract for nhl.closing_lines, the shape every sport shares; the
# NHL spread is the puck line.
CLOSING_LINE_COLUMNS = [
    "game_id",
    "season",
    "start_date",
    "source",
    "observed_at",
    "fetched_at",
    "home_spread",
    "spread_home_price",
    "spread_away_price",
    "total",
    "over_price",
    "under_price",
    "home_moneyline",
    "away_moneyline",
]


def _two_sided(price: object) -> bool:
    return price is not None and np.isfinite(price) and abs(price) >= 100


def closing_quotes(offers: pd.DataFrame, games) -> list[dict]:
    """Every quote in `offers` that could be its game's close: a two-sided
    moneyline, stamped by the provider inside the closing window and before
    the scheduled start, and fetched before the start."""
    starts = {game.game_id: game for game in games}
    lead = timedelta(minutes=CLOSING_LINE_MINUTES)
    rows = []
    quotes = partner.market_snapshot(offers[offers["game_id"].isin(starts)])
    for quote in quotes.itertuples(index=False):
        game = starts[quote.game_id]
        observed = pd.to_datetime(quote.provider_last_update, utc=True)
        fetched = pd.to_datetime(quote.fetched_at, utc=True)
        if not (
            _two_sided(quote.home_price)
            and _two_sided(quote.away_price)
            and game.start - lead <= observed < game.start
            and observed <= fetched < game.start
        ):
            continue
        rows.append(
            {
                "game_id": game.game_id,
                "season": game.season,
                "start_date": game.start,
                "source": SOURCE_PREFIX + quote.provider_key,
                "observed_at": observed.to_pydatetime(),
                "fetched_at": fetched.to_pydatetime(),
                "home_spread": quote.puck_line,
                "spread_home_price": quote.home_puck_price,
                "spread_away_price": quote.away_puck_price,
                "total": quote.total_line,
                "over_price": quote.over_price,
                "under_price": quote.under_price,
                "home_moneyline": quote.home_price,
                "away_moneyline": quote.away_price,
            }
        )
    return rows


def newer(quote: dict, than: dict | None) -> bool:
    """A later provider stamp wins; on the same stamp, the later fetch."""
    if than is None:
        return True
    key = ("observed_at", "fetched_at")
    return tuple(quote[k] for k in key) > tuple(than[k] for k in key)


class ClosingArchiver:
    def __init__(self, fetch_offers, write):
        self.fetch_offers = fetch_offers
        self.write = write
        # The latest valid quote per unstarted game, at most the tracked slate.
        self.pending: dict[str, dict] = {}
        self.next_fetch: datetime | None = None

    def capture(self, games, now: datetime) -> None:
        """Refresh the held quotes for the games dropping the puck soon and
        write the close of each game whose scheduled start has passed. `games`
        are the worker's tracked, unfinished games (game_id, season, start)."""
        games = list(games)
        tracked = {game.game_id for game in games}
        for game_id in [g for g in self.pending if g not in tracked]:
            del self.pending[game_id]
        lead = timedelta(minutes=CLOSING_LINE_MINUTES)
        imminent = [g for g in games if now < g.start <= now + lead]
        if imminent and (self.next_fetch is None or now >= self.next_fetch):
            self.next_fetch = now + timedelta(seconds=POLL_SECONDS)
            offers, _ = self.fetch_offers()
            for quote in closing_quotes(offers, imminent):
                if newer(quote, self.pending.get(quote["game_id"])):
                    self.pending[quote["game_id"]] = quote
        started = [
            g.game_id for g in games if g.start <= now and g.game_id in self.pending
        ]
        if not started:
            return
        # One attempt per game: the quotes leave memory before the write, so a
        # refused insert is logged by the caller and never retried every poll.
        closes = [self.pending.pop(game_id) for game_id in started]
        self.write(pd.DataFrame(closes, columns=CLOSING_LINE_COLUMNS))
