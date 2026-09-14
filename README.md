# momentumnhl

NHL win probabilities, goal totals, and partner-book picks from a Poisson goal
model. It is a Python port of a Google Sheets model: each team's last 25 games
at a venue, split by situation, feed a linear goal map; attack and defense
strengths are ratios to the league average; a matchup multiplies them into
expected goals for each side, and a Poisson grid turns those into win and
total probabilities. Outputs publish to the `nhl` schema of a Postgres
database that a separate web frontend ([renenunez.dev](https://renenunez.dev),
repository `momentumweb`) reads.

## How it works

1. **Windows.** MoneyPuck's team game-by-game rows are summed over each team's
   last 25 regular-season games at home and, separately, on the road, by
   situation: even strength (5v5), power play (5v4), penalty kill (4v5), and
   everything else (4v4, 3v3, 5v3, empty net). Windows end the day before the
   run and reach into the previous season, so October is never empty. A window
   under 10 games flags the team and blocks pricing of its games.
2. **Goal map.** Per situation, goals per 60 minutes is a linear function of
   ten shot-quality rates (shots, scoring-chance, high, medium and low danger
   shots and shooting percentages; the mirror set with save percentages for
   goals against), fitted once on 2021 through 2024 team-seasons and stored in
   `backend/data_static/goal_map.json`. Expected goals per game at a venue is
   the sum over situations of that rate times the situation's ice time per
   game. The fit is close to an identity on observed goals per 60, which the
   sheet also showed (R squared 0.999); replacing it with expected goals is the
   first deferred improvement.
3. **Ratings.** Attack strength is a team's expected goals for divided by the
   league average at that venue; defense strength likewise for goals against.
   Rating is (home attack + away attack) minus (home defense + away defense).
4. **Matchup.** Home expected goals = home attack of the home team times away
   defense of the away team times the league average home goals; the away
   side mirrors it. A 0 to 10 Poisson grid gives the home and away win
   probabilities with ties split evenly, fair decimal and American prices, the
   sheet's minimum acceptable price (fair decimal times 1.1), the model total,
   and over, under and push probabilities for a posted line.
5. **Prices and picks.** Lines come from the NHL's own partner sportsbook feed
   (DraftKings for the US feed, FanDuel for the Canadian feed). A moneyline is
   recommended when the model probability beats the price's break-even
   probability by at least 13 percentage points with positive expected value;
   a total when the model total differs from the posted line by at least one
   goal. Stakes are a flat one unit; the sheet's fractional Kelly (10 percent
   of full Kelly, as a share of bankroll) is published for reference. One
   decision per game and market, frozen at first publication.
6. **Grading.** Official NHL final scores settle decisions the next morning.
   A moved, postponed or cancelled game voids its decision.

## Backtest

Walk-forward replay of the 2022-23 through 2025-26 regular seasons with the
same windows and goal map, model only (no free closing-line archive exists):

| Games | Log loss | Brier | Accuracy | Home win rate | Total MAE | Total bias |
| --- | --- | --- | --- | --- | --- | --- |
| 5,212 | 0.685 | 0.245 | 57.0% | 53.8% | 1.90 | -0.16 |

A coin flip scores 0.693 log loss. The calibration bins show the expected
overconfidence of a ratio model: games priced at 74 percent for the home side
were won 67 percent of the time.

## Data sources

- MoneyPuck team game-by-game CSV (about 126 MB, downloaded once per run and
  cached as parquet by season under `backend/data/raw/moneypuck/`).
- NHL API (`api-web.nhle.com`): schedule, scores, standings and logos, and the
  partner odds feed. Official season results for the backtest are cached under
  `backend/data/raw/results/`.
- The Odds API is never called; the shared key's quota belongs to the other
  models.

## Commands

```bash
poetry install
cp .env.example .env
poetry run python -m backend ingest            # download MoneyPuck
poetry run python -m backend fit-goal-map      # refit backend/data_static/goal_map.json
poetry run python -m backend ratings           # print today's ratings
poetry run python -m backend project           # print the next week's projections
poetry run python -m backend odds              # print current partner offers
poetry run python -m backend decide            # the whole day without publishing
poetry run python -m backend backtest --publish
poetry run python -m backend daily             # the production morning run
```

Database writes are blocked unless `GITHUB_ACTIONS=true` or
`MOMENTUMNHL_DB_WRITES=1`. The daily workflow runs on a Supabase pg_cron
dispatch at 14:00 UTC from the day before opening night through April.

## Schema

`sql/001_nhl_schema.sql` is the frontend contract: `teams`, `team_ratings`,
`game_projections` (archived pregame into `forecast_snapshots` by triggers),
`market_snapshots`, `game_results`, the `live_predictions` view,
`backtest_predictions`, `recommendation_schedule`, `recommendations` with its
settlement rules, the `recommendation_performance` view, and the
`recommendation_dashboard` and `forecast_accuracy` functions.

## Deferred improvements

Renormalize or widen the Poisson grid; model overtime and the shootout
explicitly; Dixon-Coles low-score dependence; an expected-goals goal map;
shrinkage or exponential weighting instead of a hard 25-game window; goalie
awareness; puck line pricing; closing-line capture near puck drop.
