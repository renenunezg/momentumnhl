"""Acceptance gate: future outcomes, quotes and starters cannot leak backward."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from backend.model import goalies, market, signals
from backend.validation import calibrate


def test_chronological_forecasts_and_external_evidence_cutoffs():
    games, fixtures, goalies_rows = [], [], []
    for season in range(2021, 2026):
        for i in range(120):
            day = date(season, 10, 1) + timedelta(days=i * 2)
            gid = f"{season}02{i:04d}"
            fixtures.append(
                dict(
                    game_id=gid,
                    season=season,
                    game_date=day,
                    home_abbr="BOS",
                    away_abbr="TOR",
                    home_goals=3 + i % 2,
                    away_goals=5 if i % 3 == 0 else 2,
                    last_period_type="REG",
                    forecast_at=pd.Timestamp(day, tz="UTC"),
                )
            )
            for team, venue, gf, ga in (
                ("BOS", "home", 3 + i % 2, 2),
                ("TOR", "away", 2, 3 + i % 2),
            ):
                games.append(
                    dict(
                        season=season,
                        game_date=day,
                        game_id=gid,
                        team_abbr=team,
                        venue=venue,
                        gf=gf,
                        ga=ga,
                        xgf=3.1 if venue == "home" else 2.5,
                        xga=2.5 if venue == "home" else 3.1,
                        toi=3600,
                    )
                )
                goalies_rows.append(
                    dict(
                        goalie_id=team,
                        team_abbr=team,
                        game_date=day,
                        game_id=gid,
                        toi=3600,
                        xga=ga + 0.2,
                        ga=ga,
                    )
                )
    games = pd.DataFrame(games)
    fixtures = pd.DataFrame(fixtures).query("season>=2022")
    features = signals.prepare(games, fixtures)
    prediction = signals.predict(features, signals.Specification(overtime=True))
    revised = games.copy()
    revised.loc[revised.game_date >= date(2024, 10, 1), ["gf", "ga", "xgf", "xga"]] *= 5
    revised_fixtures = fixtures.copy()
    revised_fixtures.loc[revised_fixtures.season >= 2024, "home_goals"] = 20
    changed = signals.predict(
        signals.prepare(revised, revised_fixtures), signals.Specification(overtime=True)
    )
    before = prediction.game_date < date(2024, 10, 1)
    assert_frame_equal(prediction[before], changed[before])
    first_day = prediction.game_date.eq(date(2024, 10, 1))
    assert np.allclose(
        prediction.loc[first_day, "home_win_prob"],
        changed.loc[first_day, "home_win_prob"],
    )
    calibrated, fitted = calibrate(prediction)
    revised_calibrated, revised_fitted = calibrate(changed)
    original_fit = next(row for row in fitted if row["season"] == 2024)
    revised_fit = next(row for row in revised_fitted if row["season"] == 2024)
    assert original_fit["probability"]["n"] >= 200
    assert original_fit == revised_fit
    earlier = calibrated.query("season<2024")
    revised_earlier = revised_calibrated.query("season<2024")
    assert_frame_equal(earlier, revised_earlier)
    for side in ("home", "away"):
        for pool in ("venue", "pooled"):
            assert (
                features[f"{side}_{pool}_source_date"].dt.date < features.game_date
            ).all()

    slate = fixtures.tail(1).copy()
    history = pd.DataFrame(goalies_rows)
    base = goalies.forecast(history, slate)
    cutoff = slate.iloc[0].forecast_at
    confirmations = pd.DataFrame(
        [
            dict(
                game_id=slate.iloc[0].game_id,
                team_abbr="BOS",
                goalie_id="unseen",
                observed_at=cutoff + pd.Timedelta(1, unit="h"),
                source="official",
            )
        ]
    )
    assert_frame_equal(base, goalies.forecast(history, slate, confirmations))
    confirmations["observed_at"] = cutoff - pd.Timedelta(1, unit="h")
    confirmed = goalies.forecast(history, slate, confirmations)
    assert confirmed.home_goalie_confirmed.all()
    assert confirmed.home_goalie_adjustment.eq(0).all()

    forecasts = slate.assign(
        start_date=cutoff + pd.Timedelta(20, unit="h"),
        model_total=6.0,
        home_win_prob=0.55,
    )
    quotes = pd.DataFrame(
        [
            dict(
                game_id=slate.iloc[0].game_id,
                provider_key="fanduel",
                fetched_at=cutoff - pd.Timedelta(1, unit="h"),
                provider_last_update=cutoff - pd.Timedelta(2, unit="h"),
                home_price=-120.0,
                away_price=110.0,
                over_price=-110.0,
                under_price=-110.0,
                total_line=6.5,
            )
        ]
    )
    assert len(market.paired_quotes(forecasts, quotes)) == 1
    late = quotes.assign(fetched_at=cutoff + pd.Timedelta(1, unit="h"))
    assert market.paired_quotes(forecasts, late).empty
    assert len(market.paired_quotes(forecasts, late, timing="closing")) == 1
    stale = quotes.assign(provider_last_update=cutoff - pd.Timedelta(2, unit="D"))
    assert market.paired_quotes(forecasts, stale).empty

    # The complete blend evaluator must lock weights before seeing holdout
    # outcomes, even when the later season would favor the opposite choice.
    prediction_rows, quote_rows = [], []
    for season, count in ((2023, 200), (2025, 100)):
        instant = pd.Timestamp(f"{season}-10-01", tz="UTC")
        for index in range(count):
            game_id = f"{season}-{index}"
            prediction_rows.append(
                dict(
                    game_id=game_id,
                    season=season,
                    forecast_at=instant,
                    start_date=instant + pd.Timedelta(10, unit="h"),
                    home_goals=2,
                    away_goals=3,
                    model_total=8.0,
                    home_win_prob=0.8,
                )
            )
            quote_rows.append(
                dict(
                    game_id=game_id,
                    provider_key="fanduel",
                    fetched_at=instant - pd.Timedelta(1, unit="h"),
                    provider_last_update=instant - pd.Timedelta(2, unit="h"),
                    home_price=-120.0,
                    away_price=110.0,
                    over_price=-110.0,
                    under_price=-110.0,
                    total_line=5.0,
                )
            )
    sample = pd.DataFrame(prediction_rows)
    archive = pd.DataFrame(quote_rows)
    initial = market.evaluate(sample, archive)
    sample.loc[sample.season.eq(2025), "home_goals"] = 10
    changed = market.evaluate(sample, archive)
    assert initial["status"] == "evaluated"
    assert initial["selected"] == changed["selected"]
    assert initial["probability_selected"] == changed["probability_selected"]
    assert initial["selected"]["weight"] == 1
    assert initial["blend_holdout_mae"] != changed["blend_holdout_mae"]
