"""Pregame decisions at the sheet's thresholds, frozen at first publication,
and their settlement at the recorded line and price."""

import json
from datetime import timedelta

import pandas as pd

from backend.config import (
    MAX_OFFER_AGE_HOURS,
    MONEYLINE_MIN_EDGE,
    POLICY_VERSION,
    TOTAL_MIN_EDGE_GOALS,
)
from backend.model import pricing
from backend.model.poisson import total_probabilities

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
    "total_min_edge_goals": TOTAL_MIN_EDGE_GOALS,
    "tie_split": 0.5,
    "grid_max_goals": 10,
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
        "source_timestamps": json.dumps(receipts),
        "data_flags": json.dumps(flags),
        "pricing_weights": json.dumps(PRICING_WEIGHTS),
    }


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
    stale_before = decision_at - timedelta(hours=MAX_OFFER_AGE_HOURS)
    rows = []
    for projection in projections.itertuples(index=False):
        game_offers = offers[offers["game_id"].eq(projection.game_id)]
        flags = {"missing_input_count": int(projection.missing_input_count)}
        for market in MARKETS:
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
            market_offers = game_offers[game_offers["market"].eq(market)]
            market_offers = market_offers[market_offers["price"].abs() >= 100]
            if market_offers.empty:
                rows.append(
                    _no_play(
                        projection, market, "no_offer", decision_at, receipts, flags
                    )
                )
                continue
            fresh = market_offers[
                (_timestamp(market_offers["provider_last_update"]) >= stale_before)
                & (
                    _timestamp(market_offers["provider_last_update"])
                    <= _timestamp(market_offers["fetched_at"])
                )
            ]
            if fresh.empty:
                rows.append(
                    _no_play(
                        projection, market, "stale_offer", decision_at, receipts, flags
                    )
                )
                continue
            candidates = [
                _candidate(projection, offer, market)
                for offer in fresh.itertuples(index=False)
            ]
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
            rows.append(
                {
                    **_base(projection, market, decision_at, receipts, flags),
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
