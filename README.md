# momentumnhl

NHL win probabilities, goal totals, and picks priced against the league's partner sportsbooks, from a Poisson goal model.
Outputs publish to the `nhl` schema of the shared Supabase Postgres database, which the separate website repository `momentumweb` ([renenunez.dev](https://renenunez.dev)) reads.
Supabase is the only interface between the two repositories.

## How the model works

1. Each team's recent regular-season games supply goals and MoneyPuck expected goals (xG), for and against.
   Attack and defense are ratios to the league average, shrunk toward it so a short sample cannot produce an extreme rating.
   A team with too few recent games is flagged and not priced.
2. Home expected goals multiply home attack, opponent defense, and the league's home scoring rate; the away side mirrors it.
   Teams on the second night of a back-to-back are adjusted.
3. A Poisson grid over both teams' goals gives the win probability, the total, and fair prices.
   Regulation ties are split evenly between the two sides.
4. When the partner books quote a game two-sided, the published forecast is anchored to the market: the win probability starts from the de-vigged consensus, and the model total blends with the market's total.
   The goal rates are then solved so the grid reproduces that forecast, so the published probability, total, prices, picks, and in-game anchor are one forecast.
   The pure model's figures and the market inputs are stored beside it.
   Games the books do not yet quote stay pure-model until their game-day run.
5. A moneyline needs positive expected value at the quoted price; a total also needs a minimum gap between the published total and the line.
   A market not quoted two-sided is a no-play.
   Stakes are a flat one unit.
   Estimated expected value does not establish a real betting advantage.
6. Official NHL final scores settle decisions the next morning; a moved, postponed, or cancelled game voids its decision.

Rules that must hold:

- Decisions freeze at first publication, and the database rejects changes to published picks and started games.
- No lookahead: ratings, backtests, and evaluations use only data available before each game, and observations after a forecast's cutoff are ignored.
- Closing lines are a benchmark only; no forecast or decision reads one.
- Quotes must be fresh and timestamped, or independently corroborated against the book's public listing, before they can be priced.

## Live win probability and closing lines

During games, a worker turns the score, clock, and manpower into a home win probability, anchored on the frozen pregame forecast, and writes it to `nhl.live_win_probability`.
It also holds the freshest two-sided partner quote from the minutes before puck drop and writes it at the start as the game's closing line, which the database then freezes.
A game with no qualifying quote has no close.
`nhl.forecast_accuracy` compares the model with the closing line on the same games, for the backtest and the live season.

## Data sources

- [MoneyPuck](https://moneypuck.com/data.htm) team game-by-game data, including xG, cached by season.
- NHL API (`api-web.nhle.com`): schedule, scores, standings, rosters, logos, and the partner sportsbook odds (DraftKings in the US, FanDuel in Canada).
- DraftKings' public betting listing, fetched only to corroborate a quote whose provider timestamp is stale.
- A one-time cached pull of historical closing lines from ESPN, used only as the backtest benchmark.

No paid odds API is used.

## Layout

```text
backend/cli.py        python -m backend <command>
backend/etl/          MoneyPuck, NHL API, play-by-play, and live-feed ingestion
backend/model/        ratings, Poisson grid, market anchoring, pricing, in-game model
backend/odds/         partner quotes, quote verification, closing lines
backend/data_static/  team list and frozen model parameters
backend/data/         local caches (gitignored)
sql/                  ordered migrations for the nhl schema
ops/                  Supabase pg_cron dispatch jobs
tests/                production-critical tests
```

## Setup

Python 3.11 or 3.12 and [Poetry](https://python-poetry.org/).

```bash
poetry install
cp .env.example .env
poetry run pytest -q
```

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | Direct Postgres connection to the shared Supabase project |
| `MOMENTUMNHL_DB_WRITES` | Set to `1` to allow database writes from a local shell |

Core commands:

```bash
poetry run python -m backend ingest
poetry run python -m backend daily --no-publish
poetry run python -m backend grade
poetry run python -m backend backtest
poetry run python -m backend live-win-probability
```

Run `poetry run python -m backend --help` for the full list, including the research and validation commands.
Research commands produce unpublished forecasts and evaluations and never call the production publisher.

## Production

Supabase pg_cron dispatches the GitHub Actions workflows during the NHL season.
The daily workflow runs the morning pipeline (`daily`): it settles finished games, ingests data, waits for the partner feed to reach the day's slate, prices, and publishes.
A retry job resends the daily dispatch through the afternoon only when no run has happened that day, and started games are skipped, so a late run is safe.
A separate pg_cron job dispatches the live win probability workflow around games in progress.
A successful cron statement does not prove that GitHub accepted the dispatch or that the run completed, so check the Actions runs.

Database writes are blocked unless `GITHUB_ACTIONS=true` (set by CI) or `MOMENTUMNHL_DB_WRITES=1`.
`daily`, `backtest --publish`, and `live-win-probability --publish` write to production.

## Schema

The ordered `sql/*.sql` migrations define the `nhl` schema and are the contract with `momentumweb`.
This repository owns the schema.
Apply migrations in order, and apply a new one before deploying code or site pages that depend on it.
Research outputs do not alter the schema unless they are promoted into production.

## License

MIT.
