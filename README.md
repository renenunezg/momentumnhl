# momentumnhl

NHL win probabilities, goal totals, and partner-book picks from a Poisson goal model.
The production model is a Python port of a Google Sheets model: each team's last 25 games at a venue, split by situation, feed a linear goal map.
Attack and defense strengths are ratios to the league average; a matchup multiplies them into expected goals for each side.
Outputs publish to the `nhl` schema of a Postgres database that a separate web frontend ([renenunez.dev](https://renenunez.dev), repository `momentumweb`) reads.

## Production model

1. **Windows.** MoneyPuck's team game-by-game rows are summed over each team's last 25 regular-season games at home and on the road, by situation: 5v5, 5v4, 4v5, and other play.
   Windows end the day before the run and reach into the previous season.
   A window under 10 games flags the team and blocks pricing.
2. **Goal map.** Goals per 60 minutes is a linear function of ten shot-quality rates, fitted on 2021 through 2024 team-seasons and stored in `backend/data_static/goal_map.json`.
   The fit is close to an identity on observed goals; the experimental model evaluates MoneyPuck's actual xGoals instead.
3. **Ratings.** Attack is expected goals for divided by the league average at that venue; defense uses goals against.
   Rating is (home attack + away attack) minus (home defense + away defense).
4. **Matchup.** Home expected goals multiply home attack, opponent away defense, and league home goals; the away side mirrors it.
   Version 1.1 widens the Poisson grid to a tail tolerance of 1e-12 and normalizes its mass.
   Production retains equal tie allocation and rates that already include overtime; explicit regulation-to-final overtime treatment is experimental.
5. **Prices and picks.** Lines come from the NHL's partner feed: DraftKings in the US and FanDuel in Canada.
   Quotes must be dated within 24 hours.
   Moneylines require 13 percentage points of edge and positive expected value; totals require a one-goal difference.
   Stakes are flat one unit; fractional Kelly is informational.
   Decisions freeze at first publication.
6. **Grading.** Official NHL final scores, including the shootout-deciding goal, settle decisions the next morning.
   A moved, postponed or cancelled game voids its decision.

## Chronological evaluation and candidate forecasts

The original-model backtest refits its goal map using only seasons before each evaluated season, with rolling inputs strictly before each game date.
Previously published backtests used a fixed map fitted on overlapping years and should not be described as fully out of sample.
Running a local evaluation does not replace those published rows.

`validate` compares goals, MoneyPuck xG and their mixture; league shrinkage; pooled versus venue-specific windows; overtime; prior-season probability and total calibration; and inferred goalie usage.
2022-23 supplies the first out-of-season forecasts for calibration, 2023-24 and 2024-25 select specifications, and 2025-26 evaluates the locked choices.
Probability and total specifications are selected separately by log loss and MAE.
The evaluation reports coverage, calibration bins, and paired game-day bootstrap intervals against both the original model and a historical-average baseline.
No 2026-27 outcomes enter fitting or selection.
The 2025-26 season was inspected before this experiment and is a retrospective holdout, not a pristine test set.
MoneyPuck's current historical xG release can contain retrospective revisions.

The experimental goalie model uses earlier goals saved above expected, a league prior, and a mixture weighted by recent team usage.
Optional starter confirmations require `game_id`, `team_abbr`, `goalie_id`, `observed_at`, and `source`; observations after the forecast cutoff are ignored.
Postgame appearances are never treated as pregame confirmations.
Overtime experiments estimate regulation xG by prorating game-level xG to 60 minutes, then add the decisive goal for a regulation tie.
Period-specific xG would improve that approximation.

`candidate` produces unpublished forecasts using the locked selection in `backend/data_static/candidate_model.json`.
It refits calibration on earlier completed seasons and does not call the production publisher.
Its independently calibrated moneyline probability and MAE-oriented total estimate must not be passed into v1's bet-pricing calculation.
A live rollout needs explicit approval and a coherent pricing/distribution integration.

`validate-market` compares pure totals, market totals and blends on the same games.
It selects weights on earlier seasons and requires timestamped, fresh, two-sided quotes from the same provider and fetch.
Decision-time and closing quotes remain separate; closing prices cannot revise morning forecasts.
Historical quote coverage is currently insufficient to establish whether blending reduces MAE.

## Data sources

- [MoneyPuck](https://moneypuck.com/data.htm) team game-by-game CSV, including xGoals: about 126 MB, downloaded once per run and cached by season under `backend/data/raw/moneypuck/`.
- MoneyPuck's listed goalie game-by-game archives, downloaded only by the optional `ingest-goalies` command and cached under `backend/data/raw/goalies/`.
- NHL API (`api-web.nhle.com`): schedule, scores, standings, logos and partner odds.
  Historical official results are cached under `backend/data/raw/results/`.
- The Odds API is never called.

## Commands

```bash
poetry install
cp .env.example .env
poetry run python -m backend ingest
poetry run python -m backend ratings
poetry run python -m backend project
poetry run python -m backend odds
poetry run python -m backend daily --no-publish
poetry run python -m backend backtest
poetry run python -m backend ingest-goalies --seasons 2021 2022 2023 2024 2025
poetry run python -m backend validate --output /tmp/nhl-validation
poetry run python -m backend candidate --selection backend/data_static/candidate_model.json
poetry run python -m backend validate-market --predictions forecasts.parquet --snapshots quotes.parquet
```

Market evaluation needs one row per game with `game_id`, `season`, `forecast_at`, `start_date`, `model_total`, `home_win_prob`, and official `home_goals` and `away_goals`.
Quotes use the `nhl.market_snapshots` schema.
The `candidate` command accepts `--confirmations` for a timestamped starter parquet when the selected specification uses goalie adjustments.

Database writes are blocked unless `GITHUB_ACTIONS=true` or `MOMENTUMNHL_DB_WRITES=1`.
The production command is `poetry run python -m backend daily`; `backtest --publish` also writes remotely.
Supabase pg_cron dispatches GitHub Actions at 14:00 UTC on September 28-30 and daily October through April.
Python and MoneyPuck downloads run on GitHub, while Supabase stores the derived outputs.
A successful cron SQL statement does not prove that GitHub accepted the HTTP dispatch or completed the pipeline.

## Schema

`sql/001_nhl_schema.sql` is the frontend contract.
It defines teams, ratings, projections, immutable pregame snapshots, market snapshots, official results, backtest predictions, recommendations, and reporting views/functions.
Research outputs and starter inputs do not alter this schema.

## Remaining work

Repair the dispatch authorization and verify a full published-and-graded production cycle.
Collect timestamped pregame market and starter evidence before evaluating their incremental value in live conditions.
Candidate promotion, coherent calibrated bet pricing, period-specific overtime inputs, puck-line pricing and closing-line capture remain separate work.
