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
   Quotes need a provider timestamp within 24 hours or an exact, independently corroborated DraftKings public listing observed within 15 minutes of publication.
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

## Live win probability

`nhl-ingame-v1` turns the score, clock and manpower into a home win probability while a game is played.
The score margin is a Markov chain whose scoring rates are each side's pregame expected goals times league multipliers for the score state and time left; power plays and empty nets are played out on top with their own multipliers.
Regulation ties are an even split, as in the pregame model.
Multipliers were fitted on 2022-23 and 2023-24, manpower handling was kept on 2024-25 log loss, and 2025-26 scored the frozen choice: log loss 0.514 over 1,312 games against 0.516 for score and time alone.
The published multipliers in `backend/data_static/ingame_model.json` are that holdout fit (2022-23 through 2024-25).

The anchor is the published pregame forecast, so its overconfidence carries into the early game.
Compressing the pregame goal ratio by a factor fitted on earlier seasons lowered 2025-26 in-game log loss by 0.006 (95% interval 0.003 to 0.010); that is a pregame calibration question and is not applied here.
Stacked penalties are dated from the latest one, and nothing has been compared with a live market.

`live-win-probability` reads the NHL score feed once per poll for the slate, scores every game inside its puck-drop window against the forecast `nhl.game_projections` froze, and writes `nhl.live_win_probability`.
It never writes a pregame table, is read only unless `--publish` is passed, and exits when no game is left to watch.
A Supabase pg_cron job (`ops/install_live_cron.sql`) checks every five minutes from September through April and dispatches the `live win probability` workflow only while a started game has no terminal row and no worker is alive.

## Data sources

- [MoneyPuck](https://moneypuck.com/data.htm) team game-by-game CSV, including xGoals: about 126 MB, downloaded once per run and cached by season under `backend/data/raw/moneypuck/`.
- MoneyPuck's listed goalie game-by-game archives, downloaded only by the optional `ingest-goalies` command and cached under `backend/data/raw/goalies/`.
- NHL API (`api-web.nhle.com`): schedule, scores, standings, logos and partner odds.
  Historical official results are cached under `backend/data/raw/results/`.
- The Odds API is never called.

## Independent quote verification

When a future DraftKings offer has old provider metadata, the daily run makes at most one additional request to DraftKings Network's public NHL betting-splits page.
It matches the teams, scheduled date, full-game market, side, total and exact American price; the NHL partner feed remains the source of the published price.
The book's listed puck-drop time may be up to 15 minutes after the official broadcast start, but publication still stops at the earlier official time.
A response needs a valid HTTP Date and no more than five minutes of HTTP age.
The original HTML is stored by SHA-256, and matched quotes and fetch metadata are retained with the workflow artifacts.

This proves a recent observation of DraftKings' public listing, not the timestamp of its internal quote update or guaranteed bet acceptance.
The provider's original timestamp is preserved, and independently corroborated decisions carry evidence in `data_flags.quote_verification` and `source_timestamps.quote_verification`.
Python and PostgreSQL reject evidence more than 15 minutes old at decision/publication time, mismatched fields, unsupported markets and late fixtures.
A failed fetch, changed page layout or unmatched quote leaves the old offer ineligible.
FanDuel has no independent verifier and still requires its provider timestamp.
No paid odds API or additional Supabase polling is involved.

## Commands

```bash
poetry install
cp .env.example .env
poetry run python -m backend ingest
poetry run python -m backend ratings
poetry run python -m backend project
poetry run python -m backend odds
poetry run python -m backend verify-quotes
poetry run python -m backend daily --no-publish
poetry run python -m backend backtest
poetry run python -m backend ingest-play-by-play
poetry run python -m backend ingame-backtest
poetry run python -m backend live-win-probability
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
Supabase pg_cron dispatches GitHub Actions at 16:15 UTC on September 28-30 and daily October through April.
The partner feed serves the previous slate until about noon Eastern, so the run polls for up to 90 minutes for the day's slate and fails if it never arrives.
Python and MoneyPuck downloads run on GitHub, while Supabase stores the derived outputs.
A successful cron SQL statement does not prove that GitHub accepted the HTTP dispatch or completed the pipeline.

## Schema

The ordered `sql/*.sql` migrations are the frontend contract.
They define teams, ratings, projections, immutable pregame snapshots, market snapshots, official results, backtest predictions, recommendations, live win probability snapshots, and reporting views/functions.
Research outputs and starter inputs do not alter this schema.

## Remaining work

Verify a full published-and-graded production cycle after the repaired dispatch authorization.
Collect timestamped pregame market and starter evidence before evaluating their incremental value in live conditions.
Candidate promotion, coherent calibrated bet pricing, period-specific overtime inputs, puck-line pricing and closing-line capture remain separate work.


Settlement runs before pregame ingestion and includes dates from every unresolved ledger entry, even outside the routine three-day result window.
Fresh odds or MoneyPuck failures do not roll back already committed settlements.
Production workflow artifacts retain compact `forecast_replay` source bodies under `receipts/sources`, with normalized pregame windows, fitted coefficients, schedule, offers, and receipt identities.
Use `backend.replay.forecast(Path(...))` on the retained gzip body to reconstruct ratings, projections, and decisions offline with the matching source revision.
The full MoneyPuck download is identified by its digest; replay uses the sufficient normalized windows rather than treating a later download as an earlier vintage.
Backtest parquet metadata includes eligible and evaluated counts plus excluded game IDs and reasons.
Unexpected model validation failures propagate instead of silently removing games from reported performance.

The website also requires `sql/005_snapshot_lookup.sql` for its bounded record and snapshot reads.
