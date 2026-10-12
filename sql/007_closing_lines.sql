-- Closing lines as the market benchmark for forecast accuracy. They grade
-- nothing and no forecast reads one: a missing close stays null and its game
-- is left out of every market comparison.
-- The backtest carries one book's historical close per game. A live game's
-- close is the freshest two-sided partner quote archived before puck drop
-- and stamped within 15 minutes of it. The feed turns to in-play lines once a
-- game starts, so a quote stamped or fetched at or after the scheduled start
-- is never a close.
begin;

alter table nhl.backtest_predictions
  add column closing_home_price float8, add column closing_away_price float8,
  add column closing_total float8, add column closing_source text,
  add constraint backtest_closing_line check (
    (closing_home_price is null) = (closing_source is null) and
    (closing_away_price is null) = (closing_source is null) and
    (closing_total is null or closing_source is not null) and
    (closing_source is null or
     (abs(closing_home_price) >= 100 and abs(closing_home_price) < 'Infinity'::float8 and
      abs(closing_away_price) >= 100 and abs(closing_away_price) < 'Infinity'::float8)) and
    (closing_total is null or
     (closing_total > 0 and closing_total < 'Infinity'::float8
      and closing_total * 2 = round(closing_total * 2)))
  );

-- New columns go last so the existing ones keep their positions. A forecast
-- published without a market leaves its pure columns null because it is pure.
create or replace view nhl.live_predictions with (security_invoker = true) as
select r.game_id, r.season, r.game_date, r.start_date, r.home_team, r.away_team,
  r.home_goals, r.away_goals, r.last_period_type,
  p.home_lambda, p.away_lambda, p.home_win_prob, p.model_total,
  p.as_of as forecast_as_of, p.recorded_at as forecast_recorded_at, p.model_version,
  r.source, r.source_fetched_at,
  coalesce(p.pure_home_win_prob, p.home_win_prob) as pure_home_win_prob,
  coalesce(p.pure_model_total, p.model_total) as pure_model_total,
  c.home_price as closing_home_price, c.away_price as closing_away_price,
  case when c.total_line > 0 and c.total_line < 'Infinity'::float8
    then c.total_line end as closing_total,
  'nhl_partner:' || c.provider_key as closing_source,
  c.provider_last_update as closing_observed_at
from nhl.game_results r
left join lateral (
  select f.* from nhl.forecast_snapshots f
  where f.game_id = r.game_id and f.season = r.season
    and f.home_team_abbr = r.home_team_abbr
    and f.away_team_abbr = r.away_team_abbr
    and f.recorded_at < r.start_date and f.as_of < r.start_date
  order by f.recorded_at desc, f.snapshot_id desc limit 1
) p on true
left join lateral (
  select s.* from nhl.market_snapshots s
  where s.game_id = r.game_id
    and s.fetched_at < r.start_date
    and s.provider_last_update < r.start_date
    and s.provider_last_update >= r.start_date - interval '15 minutes'
    and abs(s.home_price) >= 100 and abs(s.home_price) < 'Infinity'::float8
    and abs(s.away_price) >= 100 and abs(s.away_price) < 'Infinity'::float8
  order by s.provider_last_update desc, s.fetched_at desc, s.provider_key limit 1
) c on true
where r.start_date < now();

-- Every earlier field is unchanged. The market fields compare like for like:
-- each market figure sits beside the model's figure on the same games. In the
-- backtest the model is the pure model. In the live record model_* is the
-- published forecast, which is already blended with the morning market, and
-- pure_model_* is the pure model.
create or replace function nhl.forecast_accuracy(p_source text default 'live') returns jsonb
language sql stable security invoker set search_path = pg_catalog, nhl as $$
  with selected as materialized (
    select season, home_win_prob, model_total, home_goals, away_goals,
      pure_home_win_prob, pure_model_total,
      closing_home_price, closing_away_price, closing_total
    from nhl.live_predictions where p_source = 'live' and home_win_prob is not null
    union all
    select season, home_win_prob, model_total, home_goals, away_goals,
      home_win_prob, model_total,
      closing_home_price, closing_away_price, closing_total
    from nhl.backtest_predictions where p_source = 'backtest'
  ), scored as (
    select season,
      least(greatest(home_win_prob, 1e-6), 1 - 1e-6) as p,
      least(greatest(pure_home_win_prob, 1e-6), 1 - 1e-6) as pure_p,
      -- The close with the vig removed proportionally.
      implied.home / (implied.home + implied.away) as market_p,
      (home_goals > away_goals)::int as home_won,
      model_total, pure_model_total, closing_total,
      (home_goals + away_goals)::float8 as total
    from selected cross join lateral (select
      case when closing_home_price < 0 then closing_home_price / (closing_home_price - 100)
        else 100 / (closing_home_price + 100) end as home,
      case when closing_away_price < 0 then closing_away_price / (closing_away_price - 100)
        else 100 / (closing_away_price + 100) end as away) implied
  ), slices as (
    select season, grouping(season) as is_overall, count(*) as n,
      -avg(home_won * ln(p) + (1 - home_won) * ln(1 - p)) as log_loss,
      avg((p - home_won) ^ 2) as brier,
      avg(((p > 0.5) = (home_won = 1))::int) as accuracy,
      avg(home_won) as home_win_rate,
      avg(abs(model_total - total)) as total_mae,
      avg(model_total - total) as total_bias,
      count(market_p) as market_n,
      -avg(home_won * ln(market_p) + (1 - home_won) * ln(1 - market_p)) as market_log_loss,
      -avg(home_won * ln(p) + (1 - home_won) * ln(1 - p))
        filter (where market_p is not null) as model_log_loss_on_market_games,
      -avg(home_won * ln(pure_p) + (1 - home_won) * ln(1 - pure_p))
        filter (where market_p is not null) as pure_model_log_loss_on_market_games,
      count(closing_total) as market_total_n,
      avg(abs(closing_total - total)) as market_total_mae,
      avg(abs(model_total - total))
        filter (where closing_total is not null) as model_total_mae_on_market_games,
      avg(abs(pure_model_total - total))
        filter (where closing_total is not null) as pure_model_total_mae_on_market_games
    from scored group by grouping sets ((season), ()) having count(*) > 0
  ), calibration as (
    select least(floor(p * 10), 9)::int as bin, count(*) as n,
      avg(p) as predicted, avg(home_won) as observed
    from scored group by 1 order by 1
  )
  select jsonb_build_object(
    'overall', (select to_jsonb(s) - 'season' - 'is_overall' from slices s where is_overall = 1),
    'by_season', (select coalesce(jsonb_agg(to_jsonb(s) - 'is_overall' order by s.season desc), '[]'::jsonb)
      from slices s where is_overall = 0),
    'calibration', (select coalesce(jsonb_agg(to_jsonb(c) order by c.bin), '[]'::jsonb) from calibration c),
    'missing_forecast', (select count(*) from nhl.live_predictions
      where p_source = 'live' and home_win_prob is null)
  );
$$;

notify pgrst, 'reload schema';
commit;
