"""Archives the partner quotes for games about to start. The last one stamped
before puck drop stands as the game's closing line (nhl.live_predictions
selects it); a game with no such quote has no close, and none is backfilled.
The feed turns to in-play lines once a game starts, so only unstarted games
are ever archived here."""

from datetime import datetime, timedelta

from backend.config import CLOSING_LINE_MINUTES
from backend.odds import partner

# FanDuel restamps its feed every five minutes and DraftKings hourly.
POLL_SECONDS = 60


class ClosingArchiver:
    def __init__(self, fetch_offers, write):
        self.fetch_offers = fetch_offers
        self.write = write
        self.archived: set[tuple] = set()
        self.next_fetch: datetime | None = None

    def capture(self, starts: dict[str, datetime], now: datetime) -> None:
        """Archive each new feed stamp for the games dropping the puck soon;
        `starts` maps game ids to scheduled starts."""
        lead = timedelta(minutes=CLOSING_LINE_MINUTES)
        imminent = [g for g, start in starts.items() if now < start <= now + lead]
        if not imminent or (self.next_fetch and now < self.next_fetch):
            return
        self.next_fetch = now + timedelta(seconds=POLL_SECONDS)
        offers, _ = self.fetch_offers()
        offers = offers[offers["game_id"].isin(imminent)]
        keys = ["game_id", "provider_key", "provider_last_update"]
        stamps = list(offers[keys].itertuples(index=False, name=None))
        fresh = offers[[stamp not in self.archived for stamp in stamps]]
        if fresh.empty:
            return
        self.write(partner.market_snapshot(fresh))
        self.archived.update(stamps)
