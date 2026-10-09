"""Pregame decisions, frozen at first publication, and their settlement at
the recorded line and price. Both markets price the published projection and
need it anchored to a two-sided quote."""

from datetime import timedelta

import pandas as pd

from backend.config import (
    HOME_ICE_LOGIT,
    MARKET_ANCHOR_W_MODEL,
    MAX_OFFER_AGE_HOURS,
    MONEYLINE_MIN_EDGE,
    POLICY_VERSION,
    TOTAL_MIN_EDGE_GOALS,
)
from backend.model import pricing
from backend.model.poisson import TAIL_TOLERANCE, total_probabilities
from backend.odds import verification

MARKETS = ("h2h", "totals")
RECOMMENDATION_COLUMNS = [
    "game_id",
    "market",
    "season",
    "game_date",
    "start_date",
    "home_team",
    "away_team",
    "model_version",
    "forecast_as_of",
    "missing_input_count",
    "policy_version",
    "decision_at",
    "status",
    "reason",
    "selection",
    "side",
    "point",
    "price",
    "provider",
    "provider_key",
    "market_fetched_at",
    "provider_event_id",
    "provider_start_date",
    "provider_last_update",
    "win_probability",
    "push_probability",
    "probability_edge",
    "edge_points",
    "expected_value_per_unit",
    "stake_units",
    "kelly_fraction",
    "minimum_price",
    "home_lambda",
    "away_lambda",
    "model_total",
    "market_total",
    "source_timestamps",
    "data_flags",
    "pricing_weights",
]
SETTLEMENT_COLUMNS = [
    "game_id",
    "market",
    "outcome",
    "home_goals",
    "away_goals",
    "profit_units",
    "graded_at",
    "settlement_reason",
    "result_source_at",
]
PRICING_WEIGHTS = {
    "moneyline_min_edge": MONEYLINE_MIN_EDGE,
    "market_anchor_w_model": MARKET_ANCHOR_W_MODEL,
    "home_ice_logit": HOME_ICE_LOGIT,
    "total_min_edge_goals": TOTAL_MIN_EDGE_GOALS,
    "tie_split": 0.5,
    "grid_tail_tolerance": TAIL_TOLERANCE,
}


def _timestamp(value):
    return pd.to_datetime(value, utc=True)


def _no_play(projection, market, reason, decision_at, receipts, flags) -> dict:
    return {
        **_base(projection, market, decision_at, receipts, flags),
        "status": "no_play",
        "reason": reason,
        "stake_units": 0.0,
    }


def _base(projection, market, decision_at, receipts, flags) -> dict:
    return {
        "game_id": projection.game_id,
        "market": market,
        "season": int(projection.season),
        "game_date": projection.game_date,
        "start_date": projection.start_date,
        "home_team": projection.home_team,
        "away_team": projection.away_team,
        "model_version": projection.model_version,
        "forecast_as_of": projection.as_of,
        "missing_input_count": int(projection.missing_input_count),
        "policy_version": POLICY_VERSION,
        "decision_at": decision_at,
        "home_lambda": float(projection.home_lambda),
        "away_lambda": float(projection.away_lambda),
        "model_total": float(projection.model_total),
        "source_timestamps": receipts,
        "data_flags": flags,
        "pricing_weights": PRICING_WEIGHTS,
    }


def _eligible_offers(projection, market_offers: pd.DataFrame, decision_at):
    """The offers a decision may use: quoted for this fixture, fetched by the
    decision, and either fresh or independently verified. Also returns which
    ones needed the verification."""
    stale_before = decision_at - timedelta(hours=MAX_OFFER_AGE_HOURS)
    timestamp_fresh = _timestamp(market_offers["provider_last_update"]) >= stale_before
    verified = pd.Series(False, index=market_offers.index)
    if "quote_verification" in market_offers:
        verified = market_offers.apply(
            lambda offer: verification.valid(
                offer.quote_verification, offer, projection, decision_at
            ),
            axis=1,
        )
    valid_fixture = (
        (_timestamp(market_offers["fetched_at"]) <= decision_at)
        & (
            _timestamp(market_offers["provider_start_date"])
            == _timestamp(projection.start_date)
        )
        & (
            _timestamp(market_offers["provider_last_update"])
            <= _timestamp(market_offers["fetched_at"])
        )
    )
    return market_offers[(timestamp_fresh | verified) & valid_fixture], verified


def _priced_offers(offers: pd.DataFrame, game_id, market: str) -> pd.DataFrame:
    quoted = offers[offers["game_id"].eq(game_id) & offers["market"].eq(market)]
    return quoted[quoted["price"].abs() >= 100]


def market_consensus(
    projections: pd.DataFrame, offers: pd.DataFrame, decision_at
) -> dict:
    """What the books say about each game, from the offers a pick could use:
    `home_prob` is the de-vigged home win probability averaged over every
    book quoting both moneyline sides in one fetch, and `total` the median
    line of the books quoting both sides of one total. A game or market no
    book quotes two-sided is absent."""
    decision_at = _timestamp(decision_at)
    consensus = {}
    for projection in projections.itertuples(index=False):
        home_probs, totals = [], []
        for market, first, second in (
            ("h2h", "home", "away"),
            ("totals", "over", "under"),
        ):
            quoted = _priced_offers(offers, projection.game_id, market)
            if quoted.empty:
                continue
            eligible, _ = _eligible_offers(projection, quoted, decision_at)
            for _, book in eligible.groupby(["provider_key", "fetched_at"]):
                a, b = book[book["side"].eq(first)], book[book["side"].eq(second)]
                if len(a) != 1 or len(b) != 1:
                    continue
                if market == "totals":
                    if a["point"].iloc[0] == b["point"].iloc[0]:
                        totals.append(float(a["point"].iloc[0]))
                    continue
                home_probs.append(
                    pricing.devig(
                        pricing.implied_probability(float(a["price"].iloc[0])),
                        pricing.implied_probability(float(b["price"].iloc[0])),
                    )[0]
                )
        game = {}
        if home_probs:
            game["home_prob"] = sum(home_probs) / len(home_probs)
        if totals:
            game["total"] = float(pd.Series(totals).median())
        if game:
            consensus[projection.game_id] = game
    return consensus


def _candidate(projection, offer, market) -> dict:
    """Probability, edge, EV, and stake info for one offer."""
    if market == "h2h":
        probability = float(
            projection.home_win_prob
            if offer.side == "home"
            else projection.away_win_prob
        )
        push = 0.0
        edge_points = None
        market_total = None
    else:
        probs = total_probabilities(
            projection.home_lambda, projection.away_lambda, offer.point
        )
        probability = probs[offer.side]
        push = probs["push"]
        edge_points = float(
            projection.model_total - offer.point
            if offer.side == "over"
            else offer.point - projection.model_total
        )
        market_total = float(offer.point)
    decimal = pricing.american_to_decimal(offer.price)
    return {
        "selection": {
            "home": projection.home_team,
            "away": projection.away_team,
            "over": "Over",
            "under": "Under",
        }[offer.side],
        "side": offer.side,
        "point": None if market == "h2h" else float(offer.point),
        "price": float(offer.price),
        "provider": offer.provider,
        "provider_key": offer.provider_key,
        "market_fetched_at": offer.fetched_at,
        "provider_event_id": offer.game_id,
        "provider_start_date": offer.provider_start_date,
        "provider_last_update": offer.provider_last_update,
        "win_probability": probability,
        "push_probability": push,
        "probability_edge": pricing.probability_edge(probability, offer.price, push),
        "edge_points": edge_points,
        "expected_value_per_unit": pricing.expected_value_per_unit(
            probability, offer.price, push
        ),
        "kelly_fraction": pricing.kelly_fraction(probability, decimal),
        "minimum_price": pricing.minimum_price(1 / probability)
        if probability > 0
        else None,
        "market_total": market_total,
    }


def _eligible(candidate: dict, market: str) -> bool:
    if market == "h2h":
        return (
            candidate["probability_edge"] >= MONEYLINE_MIN_EDGE
            and candidate["expected_value_per_unit"] > 0
        )
    return candidate["edge_points"] >= TOTAL_MIN_EDGE_GOALS


def decide(
    projections: pd.DataFrame,
    offers: pd.DataFrame,
    receipts: dict,
    decision_at,
) -> pd.DataFrame:
    """One row per game and market. The best eligible offer across providers
    becomes the pick; otherwise the row records why there is no play."""
    decision_at = _timestamp(decision_at)
    rows = []
    for projection in projections.itertuples(index=False):
        for market in MARKETS:
            flags = {"missing_input_count": int(projection.missing_input_count)}
            if projection.missing_input_count > 0:
                rows.append(
                    _no_play(
                        projection,
                        market,
                        "insufficient_window",
                        decision_at,
                        receipts,
                        flags,
                    )
                )
                continue
            market_offers = _priced_offers(offers, projection.game_id, market)
            if market_offers.empty:
                rows.append(
                    _no_play(
                        projection, market, "no_offer", decision_at, receipts, flags
                    )
                )
                continue
            fresh, verified = _eligible_offers(projection, market_offers, decision_at)
            if fresh.empty:
                rows.append(
                    _no_play(
                        projection, market, "stale_offer", decision_at, receipts, flags
                    )
                )
                continue
            anchor, flag = (
                (projection.market_home_prob, "market_home_probability")
                if market == "h2h"
                else (projection.market_total, "market_total_consensus")
            )
            if pd.isna(anchor):
                # The pure model alone is overconfident against a price.
                rows.append(
                    _no_play(
                        projection,
                        market,
                        "no_paired_market",
                        decision_at,
                        receipts,
                        flags,
                    )
                )
                continue
            flags = {**flags, flag: float(anchor)}
            candidates = []
            for index, offer in fresh.iterrows():
                candidate = _candidate(projection, offer, market)
                candidate["verification_evidence"] = (
                    offer.quote_verification if verified.at[index] else None
                )
                candidates.append(candidate)
            eligible = [c for c in candidates if _eligible(c, market)]
            if not eligible:
                reason = (
                    f"edge_below_{MONEYLINE_MIN_EDGE:g}"
                    if market == "h2h"
                    else f"total_within_{TOTAL_MIN_EDGE_GOALS:g}_goals"
                )
                rows.append(
                    _no_play(projection, market, reason, decision_at, receipts, flags)
                )
                continue
            key = "probability_edge" if market == "h2h" else "edge_points"
            best = max(eligible, key=lambda c: (c[key], c["price"]))
            evidence = best.pop("verification_evidence")
            decision_flags, decision_receipts = dict(flags), dict(receipts)
            if evidence is not None:
                decision_flags["quote_verification"] = evidence
                decision_receipts["quote_verification"] = {
                    k: evidence[k] for k in ("sha256", "observed_at")
                }
            rows.append(
                {
                    **_base(
                        projection,
                        market,
                        decision_at,
                        decision_receipts,
                        decision_flags,
                    ),
                    "status": "recommended",
                    "reason": "edge_gate",
                    "stake_units": 1.0,
                    **best,
                }
            )
    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)


def settle(
    pending: pd.DataFrame,
    results: pd.DataFrame,
    observation: pd.DataFrame,
    graded_at,
) -> pd.DataFrame:
    """Outcomes for pending decisions whose game is final or whose fixture
    changed. Moneylines have no tie in the NHL; totals push on the line."""
    graded_at = _timestamp(graded_at)
    finals = results.set_index("game_id")
    fixtures = observation.set_index("game_id")
    rows = []
    for pick in pending.itertuples(index=False):
        if pick.game_id not in fixtures.index:
            continue
        fixture = fixtures.loc[pick.game_id]
        changed = fixture["game_status"] in ("canceled", "cancelled", "postponed")
        changed = changed or _timestamp(fixture["start_date"]) != _timestamp(
            pick.start_date
        )
        base = {
            "game_id": pick.game_id,
            "market": pick.market,
            "graded_at": graded_at,
            "result_source_at": fixture["observed_at"],
            "home_goals": None,
            "away_goals": None,
        }
        if changed:
            rows.append(
                {
                    **base,
                    "outcome": "void" if pick.status == "recommended" else "no_play",
                    "profit_units": 0.0,
                    "settlement_reason": "schedule_change",
                }
            )
            continue
        if pick.game_id not in finals.index:
            continue
        final = finals.loc[pick.game_id]
        home, away = int(final["home_goals"]), int(final["away_goals"])
        if pick.status == "no_play":
            rows.append(
                {
                    **base,
                    "outcome": "no_play",
                    "home_goals": home,
                    "away_goals": away,
                    "profit_units": 0.0,
                    "settlement_reason": "confirmed_final",
                }
            )
            continue
        if pick.market == "h2h":
            balance = (home - away) * (1 if pick.side == "home" else -1)
        else:
            balance = (home + away - pick.point) * (1 if pick.side == "over" else -1)
        outcome = "win" if balance > 0 else "loss" if balance < 0 else "push"
        profit = {
            "win": pricing.profit_per_unit(pick.price),
            "loss": -1.0,
            "push": 0.0,
        }[outcome]
        rows.append(
            {
                **base,
                "outcome": outcome,
                "home_goals": home,
                "away_goals": away,
                "profit_units": float(profit) * float(pick.stake_units),
                "settlement_reason": "confirmed_final",
            }
        )
    return pd.DataFrame(rows, columns=SETTLEMENT_COLUMNS)
