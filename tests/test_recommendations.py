"""The sheet's play rules and the settlement arithmetic the ledger enforces."""

from datetime import date

import pandas as pd
import pytest

from backend.model.projections import PROJECTION_COLUMNS
from backend.recommendations import decide, settle

DECISION_AT = "2026-10-07T14:00:00Z"
RECEIPTS = {
    "moneypuck": {"sha256": "a" * 64, "observed_at": "2026-10-07T13:00:00Z"},
    "schedule": {"sha256": "b" * 64, "observed_at": "2026-10-07T13:30:00Z"},
    "odds": {"sha256": "c" * 64, "observed_at": "2026-10-07T13:55:00Z"},
}


def _projection(home_win=0.62, total=6.9, missing=0):
    row = dict.fromkeys(PROJECTION_COLUMNS)
    row.update(
        game_id="2026020001",
        season=2026,
        game_date=date(2026, 10, 7),
        start_date="2026-10-07T23:00:00Z",
        as_of="2026-10-07T13:58:00Z",
        model_version="nhl-poisson-v1",
        home_team_abbr="WSH",
        away_team_abbr="PIT",
        home_team="Washington Capitals",
        away_team="Pittsburgh Penguins",
        home_lambda=3.9,
        away_lambda=3.0,
        home_win_prob=home_win,
        away_win_prob=1 - home_win,
        model_total=total,
        missing_input_count=missing,
    )
    return pd.DataFrame([row])


def _offer(market, side, price, point=None, provider="DraftKings", update=None):
    return {
        "game_id": "2026020001",
        "provider": provider,
        "provider_key": provider.lower(),
        "provider_last_update": update or "2026-10-07T13:50:00Z",
        "fetched_at": "2026-10-07T13:55:00Z",
        "provider_start_date": "2026-10-07T23:00:00Z",
        "market": market,
        "side": side,
        "point": point,
        "price": price,
    }


def test_moneyline_needs_thirteen_points_over_the_book():
    offers = pd.DataFrame(
        [
            _offer("h2h", "home", -120),  # implied 54.5%, edge 7.5 points
            _offer("h2h", "home", 105, provider="FanDuel"),  # 48.8%, edge 13.2
            _offer("h2h", "away", 100),
        ]
    )
    decisions = decide(_projection(), offers, RECEIPTS, DECISION_AT)
    h2h = decisions[decisions["market"].eq("h2h")].iloc[0]
    assert h2h["status"] == "recommended"
    assert h2h["provider"] == "FanDuel" and h2h["price"] == 105
    assert h2h["probability_edge"] == pytest.approx(0.62 - 100 / 205, abs=1e-9)
    assert h2h["expected_value_per_unit"] == pytest.approx(0.271)
    assert h2h["kelly_fraction"] == pytest.approx(0.0258095238)
    assert h2h["minimum_price"] == -129
    assert h2h["stake_units"] == 1.0
    totals = decisions[decisions["market"].eq("totals")].iloc[0]
    assert totals["status"] == "no_play" and totals["reason"] == "no_offer"


def test_total_needs_a_full_goal_and_stale_feeds_are_skipped():
    offers = pd.DataFrame(
        [
            _offer("totals", "over", -110, 5.5),
            _offer("totals", "under", -110, 5.5),
            _offer("h2h", "home", -140, update="2026-10-05T13:50:00Z"),
        ]
    )
    decisions = decide(_projection(total=6.9), offers, RECEIPTS, DECISION_AT)
    totals = decisions[decisions["market"].eq("totals")].iloc[0]
    assert totals["status"] == "recommended" and totals["side"] == "over"
    assert totals["edge_points"] == pytest.approx(1.4)
    assert totals["market_total"] == 5.5
    h2h = decisions[decisions["market"].eq("h2h")].iloc[0]
    assert h2h["reason"] == "stale_offer"
    close = decide(_projection(total=6.2), offers, RECEIPTS, DECISION_AT)
    assert close[close["market"].eq("totals")].iloc[0]["reason"] == (
        "total_within_1_goals"
    )


def test_settlement_arithmetic():
    offers = pd.DataFrame(
        [
            _offer("h2h", "home", 105),
            _offer("totals", "over", -110, 6.0),
            _offer("totals", "under", -110, 6.0),
        ]
    )
    pending = decide(_projection(total=7.2), offers, RECEIPTS, DECISION_AT)
    results = pd.DataFrame(
        [{"game_id": "2026020001", "home_goals": 4, "away_goals": 2}]
    )
    observation = pd.DataFrame(
        [
            {
                "game_id": "2026020001",
                "start_date": "2026-10-07T23:00:00Z",
                "game_status": "off",
                "observed_at": "2026-10-08T12:00:00Z",
            }
        ]
    )
    settled = settle(pending, results, observation, "2026-10-08T12:05:00Z")
    by_market = settled.set_index("market")
    assert by_market.loc["h2h", "outcome"] == "win"
    assert by_market.loc["h2h", "profit_units"] == pytest.approx(1.05)
    assert by_market.loc["totals", "outcome"] == "push"
    assert by_market.loc["totals", "profit_units"] == 0
    moved = observation.assign(start_date="2026-10-08T23:00:00Z")
    voided = settle(pending, results, moved, "2026-10-08T12:05:00Z")
    assert voided["outcome"].eq("void").all()
    assert voided["settlement_reason"].eq("schedule_change").all()
