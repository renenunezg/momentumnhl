"""Independent, exact-price checks against DraftKings' public NHL offering.

The NHL partner feed remains the source of every priced offer. A sportsbook
response can only corroborate that offer; it never supplies a replacement price.
Unavailable, ambiguous, suspended or mismatched observations fail closed.
"""

import gzip
import hashlib
import json
import math
import re
import unicodedata
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

from backend.config import MAX_OFFER_AGE_HOURS, RAW_DIR
from backend.etl import store
from backend.etl.nhl_api import USER_AGENT

SOURCE_URL = (
    "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
    "?tb_edate=n7days&tb_eg=42133&tb_emt=0"
)
METHOD = "draftkings_public_listing_v1"
MAX_AGE_MINUTES = 15
MAX_HTTP_AGE_SECONDS = 300
MAX_RESPONSE_BYTES = 4_000_000


def _time(value):
    return pd.to_datetime(value, utc=True, errors="coerce")


def _name(value):
    value = unicodedata.normalize("NFKD", str(value))
    return re.sub(r"[^a-z0-9]", "", value.encode("ascii", "ignore").decode().lower())


def _team_matches(label, game, side):
    full = game[f"{side}_team"]
    abbr = game[f"{side}_abbr"]
    # DraftKings shortens the city, retaining the nickname. Match exact
    # aliases, never substrings (New York, for example, has two teams).
    nickname = {
        "VGK": "Golden Knights",
        "TOR": "Maple Leafs",
        "DET": "Red Wings",
        "CBJ": "Blue Jackets",
        "UTA": "Mammoth",
    }.get(abbr, full.split()[-1])
    city = {
        "NYR": "NY",
        "NYI": "NY",
        "LAK": "LA",
        "NJD": "NJ",
        "SJS": "SJ",
        "TBL": "TB",
    }.get(abbr, abbr)
    aliases = {full, f"{abbr} {nickname}", f"{city} {nickname}"}
    return _name(label) in {_name(a) for a in aliases}


def _http_fresh(receipt):
    observed, server = (
        _time(receipt.get("observed_at")),
        _time(receipt.get("response_date")),
    )
    try:
        age = float(receipt.get("response_age_seconds", -1))
        return (
            pd.notna(observed)
            and pd.notna(server)
            and 0 <= age <= MAX_HTTP_AGE_SECONDS
            and -5 <= (observed - server).total_seconds() <= MAX_HTTP_AGE_SECONDS
        )
    except (TypeError, ValueError):
        return False


def _link(anchor):
    if anchor is None:
        return None, None
    url = urlparse(anchor.get("href", ""))
    event = re.fullmatch(r"/event/(\d+)", url.path)
    if url.scheme != "https" or url.netloc != "sportsbook.draftkings.com" or not event:
        return None, None
    outcomes = parse_qs(url.query).get("outcomes", [])
    return event[1], outcomes[0] if len(outcomes) == 1 else None


def corroborate(html, receipt, offers, schedule):
    """Corroborate exact offers against the book's public full-game listing.

    This records a contemporaneous public observation, not an upstream quote
    update time. DOM selectors and selection IDs are grounded in the live page.
    Unknown layouts, ambiguous fixtures, or nonstandard markets are rejected.
    """
    out = offers.copy()
    out["quote_verification"] = pd.Series(None, index=out.index, dtype=object)
    if not _http_fresh(receipt) or receipt.get("source_url") != SOURCE_URL:
        return out
    soup = BeautifulSoup(html, "html.parser")
    league = soup.select_one('select[name="tb_eg"] option[selected]')
    if league is None or league.get("value") != "42133":
        return out
    observed = _time(receipt["observed_at"])
    observations = {}
    for event in soup.select(".tb-se"):
        title = event.select_one(".tb-se-title")
        if title is None:
            continue
        anchor, date_label = title.select_one("a"), title.select_one("span")
        event_id, _ = _link(anchor)
        if event_id is None or date_label is None:
            continue
        parts = anchor.get_text(strip=True).split(" @ ")
        if len(parts) != 2:
            continue
        # This first-party page displays Eastern local time without a year.
        # The official fixture must match uniquely within the next seven days.
        matches = []
        for game in schedule:
            start = _time(game["start_date"])
            if pd.isna(start) or not observed < start <= observed + timedelta(days=7):
                continue
            local = start.astimezone(ZoneInfo("America/New_York"))
            try:
                listed = datetime.strptime(
                    f"{local.year}/{date_label.get_text(strip=True)}",
                    "%Y/%m/%d, %I:%M%p",
                ).replace(tzinfo=ZoneInfo("America/New_York"))
            except ValueError:
                continue
            # The book lists puck drop ten minutes after NHL's broadcast time.
            # Keep the official, earlier timestamp as the publication deadline.
            if (
                timedelta(0) <= listed - start <= timedelta(minutes=15)
                and _team_matches(parts[0], game, "away")
                and _team_matches(parts[1], game, "home")
            ):
                matches.append((game, listed))
        if len(matches) != 1:
            continue
        game, listed = matches[0]
        for market in event.select(".tb-market-wrap > div"):
            heading = market.select_one(".tb-se-head > div")
            kind = {"Moneyline": "h2h", "Total": "totals"}.get(
                heading.get_text(strip=True) if heading else ""
            )
            rows = market.select(".tb-sodd")
            if kind is None or len(rows) != 2:
                continue
            parsed = []
            for row in rows:
                label, anchor = (
                    row.select_one(".tb-slipline"),
                    row.select_one(".tb-odd-s"),
                )
                linked_event, outcome = _link(anchor)
                if label is None or linked_event != event_id or outcome is None:
                    continue
                label = label.get_text(strip=True)
                price_text = anchor.get_text(strip=True).replace("−", "-")
                if not re.fullmatch(r"[+-]\d+", price_text):
                    continue
                price = float(price_text)
                if not math.isfinite(price) or abs(price) < 100:
                    continue
                if kind == "h2h":
                    side = next(
                        (s for s in ("home", "away") if _team_matches(label, game, s)),
                        None,
                    )
                    point = None
                    identity = re.fullmatch(r"(0ML\d+)_(1|3)", outcome)
                    if (
                        identity is None
                        or side != {"1": "home", "3": "away"}[identity[2]]
                    ):
                        continue
                else:
                    total = re.fullmatch(r"(Over|Under) (\d+(?:\.\d+)?)", label)
                    identity = re.fullmatch(r"(0OU\d+)(O|U)(\d+)_(1|3)", outcome)
                    if total is None or identity is None:
                        continue
                    side, point = total[1].lower(), float(total[2])
                    if (
                        not math.isfinite(point)
                        or point <= 0
                        or point * 2 != round(point * 2)
                        or int(identity[3]) != round(point * 100)
                        or (identity[2], identity[4])
                        != {"over": ("O", "1"), "under": ("U", "3")}[side]
                    ):
                        continue
                parsed.append((side, point, price, outcome, identity[1]))
            expected_sides = {"home", "away"} if kind == "h2h" else {"over", "under"}
            if (
                len(parsed) != 2
                or {p[0] for p in parsed} != expected_sides
                or parsed[0][1] != parsed[1][1]
                or parsed[0][4] != parsed[1][4]
            ):
                continue
            for side, point, price, outcome, market_id in parsed:
                key = (game["game_id"], kind, side, point, price)
                evidence = {
                    **receipt,
                    "method": METHOD,
                    "provider_key": "draftkings",
                    "game_id": game["game_id"],
                    "start_date": game["start_date"],
                    "market": kind,
                    "side": side,
                    "point": point,
                    "price": price,
                    "source_event_id": event_id,
                    "source_market_id": market_id,
                    "source_selection_id": outcome,
                    "source_start_label": date_label.get_text(strip=True),
                    "source_start_date": listed.isoformat(),
                }
                observations.setdefault(key, []).append(evidence)
    for index, offer in out.iterrows():
        if offer.provider_key != "draftkings":
            continue
        point = None if pd.isna(offer.point) else float(offer.point)
        key = (offer.game_id, offer.market, offer.side, point, float(offer.price))
        hits = observations.get(key, [])
        if len(hits) == 1 and _time(offer.fetched_at) <= observed:
            out.at[index, "quote_verification"] = hits[0]
    return out


def valid(evidence, offer, projection, decision_at):
    """Recheck binding and expiry immediately before creating a decision."""
    if not isinstance(evidence, dict) or not _http_fresh(evidence):
        return False
    point = None if pd.isna(offer.point) else float(offer.point)
    expected = {
        "method": METHOD,
        "provider_key": offer.provider_key,
        "game_id": projection.game_id,
        "market": offer.market,
        "side": offer.side,
        "point": point,
        "price": float(offer.price),
        "source_url": SOURCE_URL,
    }
    market_id = str(evidence.get("source_market_id", ""))
    if offer.market == "h2h" and offer.side in ("home", "away"):
        identity_valid = bool(re.fullmatch(r"0ML\d+", market_id))
        selection = market_id + ("_1" if offer.side == "home" else "_3")
    elif (
        offer.market == "totals"
        and offer.side in ("over", "under")
        and point is not None
    ):
        identity_valid = bool(re.fullmatch(r"0OU\d+", market_id))
        prefix, suffix = ("O", "_1") if offer.side == "over" else ("U", "_3")
        selection = f"{market_id}{prefix}{round(point * 100)}{suffix}"
    else:
        return False
    observed, decision = _time(evidence.get("observed_at")), _time(decision_at)
    return (
        all(evidence.get(k) == v for k, v in expected.items())
        and offer.provider_key == "draftkings"
        and _time(evidence.get("start_date")) == _time(projection.start_date)
        and _time(offer.provider_start_date) == _time(projection.start_date)
        and timedelta(0)
        <= _time(evidence.get("source_start_date")) - _time(projection.start_date)
        <= timedelta(minutes=15)
        and _time(offer.fetched_at)
        <= observed
        <= decision
        < _time(projection.start_date)
        and decision - observed <= timedelta(minutes=MAX_AGE_MINUTES)
        and bool(re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("sha256", ""))))
        and bool(re.fullmatch(r"[0-9]+", str(evidence.get("source_event_id", ""))))
        and identity_valid
        and evidence.get("source_selection_id") == selection
    )


def verify(offers, schedule):
    """At most one public-source request per run, only for stale DK offers."""
    now = datetime.now(UTC)
    stale = offers[
        offers.provider_key.eq("draftkings")
        & (
            _time(offers.provider_last_update)
            < now - timedelta(hours=MAX_OFFER_AGE_HOURS)
        )
        & (_time(offers.provider_start_date) > now)
        & offers.market.isin(("h2h", "totals"))
    ]
    report = {"source_url": SOURCE_URL, "checked": len(stale), "verified": 0}
    matches = []
    try:
        if stale.empty:
            report["status"] = "not_needed"
            return offers, report
        with requests.get(
            SOURCE_URL,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            timeout=10,
            allow_redirects=False,
            stream=True,
        ) as response:
            body = bytearray()
            for chunk in response.iter_content(65536):
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    report["status"] = "response_too_large"
                    return offers, report
            received = datetime.now(UTC)
            digest = hashlib.sha256(body).hexdigest()
            folder = RAW_DIR / "quote_verification"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{digest}.gz").write_bytes(gzip.compress(bytes(body), mtime=0))
            try:
                server_date = parsedate_to_datetime(
                    response.headers.get("Date", "")
                ).isoformat()
                age = float(response.headers.get("Age", "0"))
                if not math.isfinite(age):
                    age = None
            except (TypeError, ValueError, OverflowError):
                server_date, age = None, None
            receipt = {
                "source_url": SOURCE_URL,
                "sha256": digest,
                "observed_at": received.isoformat(),
                "response_date": server_date,
                "response_age_seconds": age,
            }
            report.update(receipt, http_status=response.status_code)
            if response.status_code != 200:
                report["status"] = "source_unavailable"
                return offers, report
            if "text/html" not in response.headers.get("Content-Type", "").lower():
                report["status"] = "invalid_response"
                return offers, report
            if not _http_fresh(receipt):
                report["status"] = "unverified_response_time"
                return offers, report
            checked = offers.copy()
            checked["quote_verification"] = corroborate(
                bytes(body), receipt, stale, schedule
            ).quote_verification.reindex(offers.index)
            matches = [
                {
                    "evidence": offer.quote_verification,
                    "partner_last_update": offer.provider_last_update,
                    "partner_fetched_at": offer.fetched_at,
                }
                for offer in checked[checked.quote_verification.notna()].itertuples()
            ]
            report["verified"] = int(checked.quote_verification.notna().sum())
            report["status"] = "verified" if report["verified"] else "no_verified_match"
            return checked, report
    except requests.RequestException as error:
        report.update(status="source_unavailable", error=type(error).__name__)
        return offers, report
    finally:
        # Store status separately from evidence; failure is never a receipt
        # that can make an offer eligible.
        store.RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
        (store.RECEIPTS_DIR / "quote_verification_matches.json").write_text(
            json.dumps(matches, indent=2) + "\n"
        )
        (store.RECEIPTS_DIR / "quote_verification_status.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
