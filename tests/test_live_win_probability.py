"""The live worker polls only while games are open and writes each final once."""

from datetime import UTC, date, datetime, timedelta

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
    games = [_game("1"), _game("2"), _game("3")]
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
    assert publisher.poll(PUCK_DROP + timedelta(hours=1))
    latest = {p["game_id"]: p for p in written}
    assert latest["1"]["home_win_probability"] > 0.8
    assert latest["2"]["abstract_state"] == "Off"

    # A sudden-death goal decides the game before the feed calls it final.
    board["1"] = LiveState("live", 2, 3, 4, "OT", 120)
    assert publisher.poll(PUCK_DROP + timedelta(hours=2, minutes=40))
    latest = {p["game_id"]: p for p in written}
    assert latest["1"]["home_win_probability"] == 0.0

    # A game that never starts is closed along with the final.
    board["1"] = LiveState("final", 2, 3)
    assert not publisher.poll(PUCK_DROP + timedelta(hours=3, minutes=1))
    latest = {p["game_id"]: p for p in written}
    assert latest["1"]["abstract_state"] == "Final"
    assert [p["s"] for p in latest["1"]["history"]] == [0, 2400, 3600]
    assert latest["3"]["abstract_state"] == "Off"

    spent, rows = len(calls), len(written)
    assert not publisher.poll(PUCK_DROP + timedelta(hours=3, minutes=2))
    assert (len(calls), len(written)) == (spent, rows)


def test_intermission_clock_is_not_time_left_in_the_period():
    state = parse_game(
        {
            "id": 2026020036,
            "gameState": "LIVE",
            "gameScheduleState": "OK",
            "homeTeam": {"score": 1},
            "awayTeam": {"score": 0},
            "periodDescriptor": {"number": 1, "periodType": "REG"},
            "clock": {"secondsRemaining": 1043, "inIntermission": True},
        }
    )
    assert (state.period, state.clock_seconds, state.intermission) == (1, 0, True)
