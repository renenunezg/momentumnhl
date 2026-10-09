"""The sheet's play rules and the settlement arithmetic the ledger enforces."""

import math
from datetime import date

import pandas as pd
import pytest

from backend.model.projections import PROJECTION_COLUMNS, anchor_to_market
from backend.recommendations import decide, market_consensus, settle

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
        pure_home_win_prob=home_win,
        pure_model_total=total,
    )
    return pd.DataFrame([row])


def _decide(projection, offers):
    """The pipeline's order: anchor the projection to the quotes, then decide."""
    anchored = anchor_to_market(
        projection, market_consensus(projection, offers, DECISION_AT)
    )
    return anchored, decide(anchored, offers, RECEIPTS, DECISION_AT)


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


def test_moneyline_prices_the_market_anchor_at_four_and_a_half_points():
    offers = pd.DataFrame(
        [
            _offer("h2h", "home", -120),  # implied 54.5%, edge 3.8 points
            _offer("h2h", "home", 105, provider="FanDuel"),  # 48.8%, edge 9.6
            _offer("h2h", "away", 100),
        ]
    )
    published, decisions = _decide(_projection(), offers)
    published = published.iloc[0]
    h2h = decisions[decisions["market"].eq("h2h")].iloc[0]
    # DraftKings is the only book quoting both sides, so it is the consensus:
    # the model's 62% plus home ice, blended halfway with the de-vigged 52.2%.
    market = (120 / 220) / (120 / 220 + 0.5)
    blended = 0.5 * (math.log(0.62 / 0.38) + 0.10) + 0.5 * math.log(
        market / (1 - market)
    )
    probability = 1 / (1 + math.exp(-blended))
    assert h2h["status"] == "recommended"
    assert h2h["provider"] == "FanDuel" and h2h["price"] == 105
    assert h2h["win_probability"] == pytest.approx(probability, abs=1e-9)
    # The published row and the frozen pick carry one forecast: the goal
    # split moved to the anchored probability with the total unchanged.
    assert published.home_win_prob == h2h["win_probability"]
    assert published.pure_home_win_prob == 0.62
    assert (published.home_lambda, published.away_lambda) == (
        h2h["home_lambda"],
        h2h["away_lambda"],
    )
    assert published.home_lambda + published.away_lambda == pytest.approx(6.9)
    assert published.model_total == 6.9
    assert h2h["probability_edge"] == pytest.approx(probability - 100 / 205, abs=1e-9)
    assert h2h["expected_value_per_unit"] == pytest.approx(2.05 * probability - 1)
    assert h2h["minimum_price"] == -113
    assert h2h["stake_units"] == 1.0
    assert h2h["data_flags"]["market_home_probability"] == pytest.approx(market)
    totals = decisions[decisions["market"].eq("totals")].iloc[0]
    assert totals["status"] == "no_play" and totals["reason"] == "no_offer"
    assert "market_home_probability" not in totals["data_flags"]
    # Without a two-sided quote there is no market to anchor on, so no pick.
    _, one_sided = _decide(_projection(), offers.iloc[[1]])
    assert one_sided[one_sided["market"].eq("h2h")].iloc[0]["reason"] == (
        "no_paired_market"
    )


def test_total_needs_half_a_goal_on_the_blend_and_stale_feeds_are_skipped():
    offers = pd.DataFrame(
        [
            _offer("totals", "over", -110, 5.5),
            _offer("totals", "under", -110, 5.5),
            _offer("h2h", "home", -140, update="2026-10-05T13:50:00Z"),
        ]
    )
    published, decisions = _decide(_projection(total=6.9), offers)
    totals = decisions[decisions["market"].eq("totals")].iloc[0]
    # The published total is halfway from the model's 6.9 to the posted 5.5,
    # and the frozen pick carries that same total.
    assert published.iloc[0].model_total == pytest.approx(6.2)
    assert totals["model_total"] == published.iloc[0].model_total
    assert published.iloc[0].pure_model_total == 6.9
    assert totals["status"] == "recommended" and totals["side"] == "over"
    assert totals["edge_points"] == pytest.approx(0.7)
    assert totals["market_total"] == 5.5
    h2h = decisions[decisions["market"].eq("h2h")].iloc[0]
    assert h2h["reason"] == "stale_offer"
    _, close = _decide(_projection(total=6.2), offers)
    assert close[close["market"].eq("totals")].iloc[0]["reason"] == (
        "total_within_0.5_goals"
    )


def test_settlement_arithmetic():
    offers = pd.DataFrame(
        [
            _offer("h2h", "home", 105),
            _offer("h2h", "away", -125),
            _offer("totals", "over", -110, 6.0),
            _offer("totals", "under", -110, 6.0),
        ]
    )
    _, pending = _decide(_projection(total=7.2), offers)
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
