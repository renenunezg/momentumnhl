"""The live worker polls only while games are open and writes each final once."""

from datetime import UTC, date, datetime, timedelta

import pytest

from backend.etl import live_feed, nhl_api
from backend.etl.live_feed import LiveState, parse_game
from backend.live_publish import Game, LivePublisher
from backend.model import ingame

PUCK_DROP = datetime(2026, 10, 4, 23, 0, tzinfo=UTC)


def _game(game_id: str) -> Game:
    return Game(
        game_id=game_id,
        season=2026,
        game_date=date(2026, 10, 4),
        start=PUCK_DROP,
        home_team="Home",
        away_team="Away",
        home_lambda=3.0,
        away_lambda=3.0,
        as_of=PUCK_DROP - timedelta(hours=6),
        model_version="test",
    )


def test_polling_starts_at_puck_drop_and_each_game_closes_once():
    games = [_game("1"), _game("2"), _game("3"), _game("4")]
    board, calls, written = {}, [], []

    def fetch_states(days):
        calls.append(days)
        return dict(board)

    publisher = LivePublisher(
        ingame.neutral(),
        load_games=lambda start, end: games,
        load_saved=lambda game_ids: {},
        fetch_states=fetch_states,
        write=written.extend,
    )

    # Inside the lead window the database alone answers; the feed is untouched.
    assert publisher.poll(PUCK_DROP - timedelta(minutes=10))
    assert calls == [] and {p["abstract_state"] for p in written} == {"Pre"}

    # A break reports the period that just ended with its clock at zero.
    board["1"] = LiveState("live", 2, 0, 2, "REG", 0, intermission=True)
    board["2"] = LiveState("off")
    board["3"] = LiveState("scheduled")
    # An unreadable row leaves only its own game unscored.
    board["4"] = LiveState("invalid", problem="bad row")
    assert publisher.poll(PUCK_DROP + timedelta(hours=1))
    latest = {p["game_id"]: p for p in written}
    assert latest["1"]["home_win_probability"] > 0.8
    assert latest["2"]["abstract_state"] == "Off"
    assert latest["4"]["home_win_probability"] is None
    assert latest["4"]["unavailable_reason"]

    # A sudden-death goal decides the game before the feed calls it final.
    board["1"] = LiveState("live", 2, 3, 4, "OT", 120)
    assert publisher.poll(PUCK_DROP + timedelta(hours=2, minutes=40))
    latest = {p["game_id"]: p for p in written}
    assert latest["1"]["home_win_probability"] == 0.0

    # A game that never starts is closed along with the final.
    board["1"] = LiveState("final", 2, 3)
    board["4"] = LiveState("final", 1, 0)
    assert not publisher.poll(PUCK_DROP + timedelta(hours=3, minutes=1))
    latest = {p["game_id"]: p for p in written}
    assert latest["1"]["abstract_state"] == "Final"
    assert [p["s"] for p in latest["1"]["history"]] == [0, 2400, 3600]
    assert latest["3"]["abstract_state"] == "Off"

    spent, rows = len(calls), len(written)
    assert not publisher.poll(PUCK_DROP + timedelta(hours=3, minutes=2))
    assert (len(calls), len(written)) == (spent, rows)


def test_score_feed_rows_parse_one_game_at_a_time(monkeypatch):
    row = {
        "id": 2026020036,
        "gameState": "LIVE",
        "gameScheduleState": "OK",
        "homeTeam": {"score": 1},
        "awayTeam": {"score": 0},
        "periodDescriptor": {"number": 1, "periodType": "REG"},
        "clock": {"secondsRemaining": 1043, "inIntermission": True},
        "situation": {"situationCode": "1451"},
    }
    # The intermission clock is not time left in the period.
    state = parse_game(row)
    assert (state.period, state.clock_seconds, state.intermission) == (1, 0, True)

    # An unreadable row marks only its own game invalid.
    bad = {**row, "id": 2026020037, "situation": {"situationCode": "1x51"}}
    with pytest.raises(ValueError, match="situation code"):
        parse_game(bad)
    monkeypatch.setattr(nhl_api, "live_scores", lambda day: [row, bad])
    states = live_feed.fetch_states({date(2026, 10, 4)})
    assert states["2026020037"].status == "invalid"
    assert states["2026020036"].situation_code == "1451"
