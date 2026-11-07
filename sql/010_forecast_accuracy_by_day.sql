-- The live record gains by_day: graded games and correct winner calls per
-- game date, so the site can draw trailing-window accuracy the way the MLB
-- page does from its daily evaluations. Counts rather than rates, so a window
-- pools its games instead of averaging daily percentages. The backtest
-- carries no dates here and returns an empty series. Every earlier field is
-- unchanged.
begin;

create or replace function nhl.forecast_accuracy(p_source text default 'live') returns jsonb
language sql stable security invoker set search_path = pg_catalog, nhl as $$
  with selected as materialized (
    select season, game_date, home_win_prob, model_total, home_goals, away_goals,
      pure_home_win_prob, pure_model_total,
      closing_home_price, closing_away_price, closing_total
    from nhl.live_predictions where p_source = 'live' and home_win_prob is not null
    union all
    select season, null::date, home_win_prob, model_total, home_goals, away_goals,
      home_win_prob, model_total,
      closing_home_price, closing_away_price, closing_total
    from nhl.backtest_predictions where p_source = 'backtest'
  ), scored as (
    select season, game_date,
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
  ), by_day as (
    select game_date as date, count(*) as n,
      sum(((p > 0.5) = (home_won = 1))::int) as correct
    from scored where game_date is not null group by game_date
  )
  select jsonb_build_object(
    'overall', (select to_jsonb(s) - 'season' - 'is_overall' from slices s where is_overall = 1),
    'by_season', (select coalesce(jsonb_agg(to_jsonb(s) - 'is_overall' order by s.season desc), '[]'::jsonb)
      from slices s where is_overall = 0),
    'calibration', (select coalesce(jsonb_agg(to_jsonb(c) order by c.bin), '[]'::jsonb) from calibration c),
    'by_day', (select coalesce(jsonb_agg(to_jsonb(d) order by d.date), '[]'::jsonb) from by_day d),
    'missing_forecast', (select count(*) from nhl.live_predictions
      where p_source = 'live' and home_win_prob is null)
  );
$$;

notify pgrst, 'reload schema';
commit;
