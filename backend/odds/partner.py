"""Partner sportsbook feeds turned into one offer per (game, provider, market,
side). The feed carries a single timestamp for the slate, so every offer from
one fetch shares provider_last_update."""

import pandas as pd

OFFER_COLUMNS = [
    "game_id",
    "provider",
    "provider_key",
    "provider_last_update",
    "fetched_at",
    "provider_start_date",
    "market",
    "side",
    "point",
    "price",
]
SNAPSHOT_COLUMNS = [
    "game_id",
    "provider_key",
    "fetched_at",
    "provider_last_update",
    "home_price",
    "away_price",
    "total_line",
    "over_price",
    "under_price",
    "puck_line",
    "home_puck_price",
    "away_puck_price",
    "quote_verifications",
]


def _parse(entry: dict, team_side: str) -> dict | None:
    description = entry.get("description")
    value = entry.get("value")
    qualifier = entry.get("qualifier") or ""
    if value is None:
        return None
    if description == "MONEY_LINE_2_WAY":
        return {"market": "h2h", "side": team_side, "point": None, "price": value}
    if description == "OVER_UNDER" and qualifier[:1] in ("O", "U"):
        return {
            "market": "totals",
            "side": "over" if qualifier[0] == "O" else "under",
            "point": float(qualifier[1:]),
            "price": value,
        }
    if description == "PUCK_LINE" and qualifier:
        return {
            "market": "puck",
            "side": team_side,
            "point": float(qualifier),
            "price": value,
        }
    return None


def offers(feed: dict) -> pd.DataFrame:
    rows = []
    for game in feed["games"]:
        for team_side in ("home", "away"):
            for entry in game[team_side]:
                parsed = _parse(entry, team_side)
                if parsed is None:
                    continue
                rows.append(
                    {
                        "game_id": game["game_id"],
                        "provider": feed["provider"],
                        "provider_key": feed["provider_key"],
                        "provider_last_update": feed["last_update"],
                        "fetched_at": feed["fetched_at"],
                        "provider_start_date": game["start_date"],
                        **parsed,
                    }
                )
    frame = pd.DataFrame(rows, columns=OFFER_COLUMNS)
    # A totals line is one number per game; keep the over and under from the
    # same posted line only.
    return frame.drop_duplicates(["game_id", "provider_key", "market", "side"])


def market_snapshot(all_offers: pd.DataFrame) -> pd.DataFrame:
    """One archive row per (game, provider, fetch) with the posted lines."""
    rows = []
    keys = ["game_id", "provider_key", "fetched_at"]
    for (game_id, provider_key, fetched_at), group in all_offers.groupby(keys):

        def price(market, side):
            hit = group[group["market"].eq(market) & group["side"].eq(side)]
            return None if hit.empty else float(hit["price"].iloc[0])

        def point(market, side):
            hit = group[group["market"].eq(market) & group["side"].eq(side)]
            return None if hit.empty else float(hit["point"].iloc[0])

        rows.append(
            {
                "game_id": game_id,
                "provider_key": provider_key,
                "fetched_at": fetched_at,
                "provider_last_update": group["provider_last_update"].iloc[0],
                "home_price": price("h2h", "home"),
                "away_price": price("h2h", "away"),
                "total_line": point("totals", "over"),
                "over_price": price("totals", "over"),
                "under_price": price("totals", "under"),
                "puck_line": point("puck", "home"),
                "home_puck_price": price("puck", "home"),
                "away_puck_price": price("puck", "away"),
                "quote_verifications": {
                    f"{offer.market}_{offer.side}": offer.quote_verification
                    for offer in group.itertuples(index=False)
                    if isinstance(getattr(offer, "quote_verification", None), dict)
                },
            }
        )
    return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
