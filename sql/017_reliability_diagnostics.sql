-- The performance page becomes a reliability check. forecast_accuracy adds
-- the close's Brier score and winner accuracy beside the forecast's on the
-- same games, one paired standard error for each gap to the close, and how
-- often favorites won by how strong they were; recommendation_dashboard adds
-- the settled moneyline record by the price taken, per pick rule, against
-- the wins each pick's probability expected. Every earlier field is
-- unchanged. Nothing here grades a pick or feeds a forecast.
begin;

create or replace function nhl.forecast_accuracy(p_source text default 'live') returns jsonb
language sql stable security invoker set search_path = pg_catalog, nhl as $$
  with selected as materialized (
    select l.season, l.game_date, l.home_win_prob, l.model_total, l.home_goals, l.away_goals,
      l.pure_home_win_prob, l.pure_model_total,
      l.closing_home_price, l.closing_away_price, l.closing_total,
      c.over_price as closing_over_price, c.under_price as closing_under_price
    from nhl.live_predictions l
    left join nhl.closing_lines c on c.game_id = l.game_id
    where p_source = 'live' and l.home_win_prob is not null
    union all
    -- The backtest close carries no over and under prices.
    select season, game_date, home_win_prob, model_total, home_goals, away_goals,
      home_win_prob, model_total,
      closing_home_price, closing_away_price, closing_total, null, null
    from nhl.backtest_predictions where p_source = 'backtest'
  ), scored as (
    select s.season, s.game_date,
      least(greatest(s.home_win_prob, 1e-6), 1 - 1e-6) as p,
      least(greatest(s.pure_home_win_prob, 1e-6), 1 - 1e-6) as pure_p,
      -- The close with the vig removed proportionally.
      implied.home / (implied.home + implied.away) as market_p,
      (s.home_goals > s.away_goals)::int as home_won,
      s.model_total, s.pure_model_total, s.closing_total,
      (s.home_goals + s.away_goals)::float8 as total,
      -- Over or under the closing total; a push is null and scores nothing,
      -- as a two-way price excludes it.
      case when s.home_goals + s.away_goals > s.closing_total then 1
           when s.home_goals + s.away_goals < s.closing_total then 0 end as went_over,
      case when s.closing_total is not null then
        least(greatest((1 - goals.at_or_under) / nullif(1 - goals.push, 0), 1e-6), 1 - 1e-6)
      end as over_p,
      implied.over_side / (implied.over_side + implied.under_side) as market_over_p
    from selected s
    cross join lateral (select
      case when s.closing_home_price < 0 then s.closing_home_price / (s.closing_home_price - 100)
        else 100 / (s.closing_home_price + 100) end as home,
      case when s.closing_away_price < 0 then s.closing_away_price / (s.closing_away_price - 100)
        else 100 / (s.closing_away_price + 100) end as away,
      case when s.closing_over_price < 0 then s.closing_over_price / (s.closing_over_price - 100)
        else 100 / (s.closing_over_price + 100) end as over_side,
      case when s.closing_under_price < 0 then s.closing_under_price / (s.closing_under_price - 100)
        else 100 / (s.closing_under_price + 100) end as under_side) implied
    -- The published total is the sum of the two goal rates, so total goals
    -- are Poisson with that mean: the distribution picks are priced from
    -- (backend/model/poisson.py total_probabilities, without overtime).
    cross join lateral (
      select coalesce(sum(mass), 0) as at_or_under,
        coalesce(sum(mass) filter (where k = s.closing_total), 0) as push
      from (
        select k, exp(-s.model_total) * power(s.model_total, k) / factorial(k)::float8 as mass
        from generate_series(0, floor(s.closing_total)::int) k
      ) pmf
    ) goals
  ), losses as (
    -- Each game's own loss under the forecast, the pure model, and the close,
    -- so a gap to the market has a paired standard error.
    select scored.*,
      -(home_won * ln(p) + (1 - home_won) * ln(1 - p)) as ll,
      -(home_won * ln(pure_p) + (1 - home_won) * ln(1 - pure_p)) as pure_ll,
      -(home_won * ln(market_p) + (1 - home_won) * ln(1 - market_p)) as market_ll,
      (p - home_won) ^ 2 as sq, (market_p - home_won) ^ 2 as market_sq,
      abs(model_total - total) - abs(closing_total - total) as total_gap
    from scored
  ), slices as (
    select season, grouping(season) as is_overall, count(*) as n,
      -avg(home_won * ln(p) + (1 - home_won) * ln(1 - p)) as log_loss,
      avg((p - home_won) ^ 2) as brier,
      avg(((p > 0.5) = (home_won = 1))::int) as accuracy,
      avg(home_won) as home_win_rate,
      avg(abs(model_total - total)) as total_mae,
      avg(model_total - total) as total_bias,
      sqrt(avg((model_total - total) ^ 2)) as total_rmse,
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
        filter (where closing_total is not null) as pure_model_total_mae_on_market_games,
      count(went_over) as over_under_n,
      avg(went_over) as over_rate,
      -avg(went_over * ln(over_p) + (1 - went_over) * ln(1 - over_p)) as over_under_log_loss,
      count(went_over) filter (where market_over_p is not null) as market_over_under_n,
      -avg(went_over * ln(market_over_p) + (1 - went_over) * ln(1 - market_over_p))
        as market_over_under_log_loss,
      -avg(went_over * ln(over_p) + (1 - went_over) * ln(1 - over_p))
        filter (where market_over_p is not null) as model_over_under_log_loss_on_market_games,
      avg(market_sq) as market_brier,
      avg(sq) filter (where market_p is not null) as model_brier_on_market_games,
      avg(((market_p > 0.5) = (home_won = 1))::int) as market_accuracy,
      avg(((p > 0.5) = (home_won = 1))::int)
        filter (where market_p is not null) as model_accuracy_on_market_games,
      -- One standard error of each gap to the close, paired by game.
      stddev_samp(ll - market_ll) / sqrt(nullif(count(market_p), 0)) as log_loss_gap_se,
      stddev_samp(pure_ll - market_ll) / sqrt(nullif(count(market_p), 0)) as pure_log_loss_gap_se,
      stddev_samp(sq - market_sq) / sqrt(nullif(count(market_p), 0)) as brier_gap_se,
      stddev_samp(total_gap) / sqrt(nullif(count(total_gap), 0)) as total_mae_gap_se
    from losses group by grouping sets ((season), ()) having count(*) > 0
  ), calibration as (
    select least(floor(p * 10), 9)::int as bin, count(*) as n,
      avg(p) as predicted, avg(home_won) as observed
    from scored group by 1 order by 1
  ), favorites as (
    -- How often the favorite won against how often it was expected to, by
    -- how strong a favorite it was: the forecast's on every game, the
    -- close's on the games that have one. Bin b covers 50 + 5b percent up
    -- to the next bin; the last holds 75 percent and above.
    select side.source, least(floor((side.favorite - 0.5) / 0.05), 5)::int as bin, count(*) as n,
      avg(side.favorite) as predicted, avg(side.won) as observed
    from scored
    cross join lateral (values
      ('model', greatest(p, 1 - p), ((p > 0.5) = (home_won = 1))::int),
      ('market', greatest(market_p, 1 - market_p), ((market_p > 0.5) = (home_won = 1))::int)
    ) side(source, favorite, won)
    where side.favorite is not null
    group by side.source, 2
  ), series as (
    -- Sums rather than rates, so a window pools its games instead of
    -- averaging daily figures. The backtest is drawn by month only.
    select period.kind, period.date, count(*) as n,
      sum(((p > 0.5) = (home_won = 1))::int) as correct,
      -sum(home_won * ln(p) + (1 - home_won) * ln(1 - p)) as log_loss_sum,
      -- The market's figures cover only the games with a recorded close.
      count(market_p) as market_n,
      coalesce(sum(((market_p > 0.5) = (home_won = 1))::int), 0) as market_correct,
      coalesce(-sum(home_won * ln(market_p) + (1 - home_won) * ln(1 - market_p)), 0)
        as market_log_loss_sum,
      sum(abs(model_total - total)) as total_abs_error_sum,
      sum(model_total - total) as total_error_sum,
      count(closing_total) as market_total_n,
      coalesce(sum(abs(closing_total - total)), 0) as market_total_abs_error_sum,
      coalesce(sum(closing_total - total), 0) as market_total_error_sum
    from scored
    cross join lateral (values
      ('day', game_date), ('month', date_trunc('month', game_date)::date)) period(kind, date)
    where period.kind = 'month' or p_source = 'live'
    group by period.kind, period.date
  ), total_errors as (
    -- Predicted minus actual goals; bin b holds errors from b up to b + 1.
    select floor(model_total - total)::int as bin, count(*) as n from scored group by 1
  )
  select jsonb_build_object(
    'overall', (select to_jsonb(s) - 'season' - 'is_overall' from slices s where is_overall = 1),
    'by_season', (select coalesce(jsonb_agg(to_jsonb(s) - 'is_overall' order by s.season desc), '[]'::jsonb)
      from slices s where is_overall = 0),
    'calibration', (select coalesce(jsonb_agg(to_jsonb(c) order by c.bin), '[]'::jsonb) from calibration c),
    'by_day', (select coalesce(jsonb_agg(to_jsonb(d) - 'kind' order by d.date), '[]'::jsonb)
      from series d where d.kind = 'day'),
    'by_month', (select coalesce(jsonb_agg(to_jsonb(d) - 'kind' order by d.date), '[]'::jsonb)
      from series d where d.kind = 'month'),
    'favorites', (select coalesce(jsonb_agg(to_jsonb(f) order by f.source, f.bin), '[]'::jsonb) from favorites f),
    'total_errors', (select coalesce(jsonb_agg(to_jsonb(e) order by e.bin), '[]'::jsonb) from total_errors e),
    'missing_forecast', (select count(*) from nhl.live_predictions
      where p_source = 'live' and home_win_prob is null)
  );
$$;

create or replace function nhl.recommendation_dashboard(
  p_season integer default null, p_market text default 'all', p_from timestamptz default null
) returns jsonb language sql stable security invoker set search_path = pg_catalog, nhl as $$
  with settled as materialized (
    select r.market, r.game_date, r.side, r.point, r.stake_units, r.profit_units, r.outcome,
      case when r.price > 0 then r.price / 100 + 1 else 100 / abs(r.price) + 1 end as taken,
      c.total as closing_total,
      case r.side when 'home' then c.home_moneyline when 'away' then c.away_moneyline
        when 'over' then c.over_price when 'under' then c.under_price end as closing_price,
      case r.side when 'home' then c.away_moneyline when 'away' then c.home_moneyline
        when 'over' then c.under_price when 'under' then c.over_price end as closing_other_price
    from nhl.recommendations r
    left join nhl.closing_lines c on c.game_id = r.game_id
    where r.status = 'recommended' and r.outcome in ('win', 'loss', 'push')
      and (p_season is null or r.season = p_season)
      and (p_market = 'all' or r.market = p_market)
      and (p_from is null or r.decision_at >= p_from)
  ), by_day as (
    select game_date as date, market, count(*) as picks,
      count(*) filter (where outcome = 'win') as wins,
      count(*) filter (where outcome = 'loss') as losses,
      count(*) filter (where outcome = 'push') as pushes,
      sum(stake_units) as staked_units, sum(profit_units) as profit_units
    from settled group by game_date, market
  ), roi_error as (
    -- One standard error of the return per unit staked.
    select coalesce(market, 'all') as market, count(*) as settled,
      stddev_samp(profit_units / stake_units) / sqrt(count(*)) as roi_se
    from settled group by grouping sets ((market), ())
  ), against_close as (
    -- Only picks whose game has a two-sided close on the pick's market.
    select s.market, versus.taken_beats_close,
      s.taken * versus.fair - 1 as value,
      case s.side when 'over' then s.closing_total - s.point
        when 'under' then s.point - s.closing_total end as line_move
    from settled s
    cross join lateral (select
      case when s.closing_price > 0 then s.closing_price / 100 + 1
        else 100 / abs(s.closing_price) + 1 end as side_decimal,
      case when s.closing_other_price > 0 then s.closing_other_price / 100 + 1
        else 100 / abs(s.closing_other_price) + 1 end as other_decimal) decimals
    cross join lateral (select
      s.taken > decimals.side_decimal as taken_beats_close,
      -- The pick's side at the close with the vig removed proportionally.
      (1 / decimals.side_decimal) / (1 / decimals.side_decimal + 1 / decimals.other_decimal) as fair
    ) versus
    where s.closing_price is not null and s.closing_other_price is not null
      and (s.market = 'h2h' or s.closing_total is not null)
  ), closing as (
    -- A moneyline beats the close on price. A total beats it when the line
    -- moved toward the pick, or held and the price shortened.
    select market, count(*) as picks,
      count(*) filter (where case when market = 'h2h' then taken_beats_close
        else line_move > 0 or (line_move = 0 and taken_beats_close) end) as beat,
      avg(value) filter (where market = 'h2h') as average_value,
      avg(line_move) filter (where market = 'totals') as average_line_move
    from against_close group by market
  ), by_price as (
    -- Settled moneyline picks by the price taken, under the rule that made
    -- them: wins against the wins the pick's own probability expected.
    select policy_version as rule, bucket.ord, bucket.label, count(*) as picks,
      count(*) filter (where outcome = 'win') as wins,
      count(*) filter (where outcome = 'loss') as losses,
      sum(win_probability) as expected_wins,
      sum(stake_units) as staked_units, sum(profit_units) as profit_units,
      avg(probability_edge) as average_edge
    from nhl.recommendations
    cross join lateral (select * from (values
      (1, '-250 or shorter', price <= -250), (2, '-249 to -150', price between -249 and -150),
      (3, '-149 to -105', price between -149 and -105), (4, '-104 to +104', price between -104 and 104),
      (5, '+105 or longer', price >= 105)) v(ord, label, matches) where matches) bucket
    where status = 'recommended' and market = 'h2h' and outcome in ('win', 'loss')
      and (p_season is null or season = p_season)
      and (p_market in ('all', 'h2h'))
      and (p_from is null or decision_at >= p_from)
    group by policy_version, bucket.ord, bucket.label
  ), by_rule as (
    -- Every recommended prediction under the pick rule that made it, pending
    -- included. The rules priced expected value differently, so an average
    -- across them is not one expectation; each rule's stands beside its ROI.
    select policy_version as rule,
      case when grouping(market) = 1 then 'all' else market end as market,
      count(*) as picks,
      count(*) filter (where outcome = 'win') as wins,
      count(*) filter (where outcome = 'loss') as losses,
      count(*) filter (where outcome = 'push') as pushes,
      count(*) filter (where outcome = 'pending') as pending,
      coalesce(sum(stake_units) filter (where outcome in ('win', 'loss', 'push')), 0) as staked_units,
      coalesce(sum(profit_units) filter (where outcome in ('win', 'loss', 'push')), 0) as profit_units,
      sum(profit_units) filter (where outcome in ('win', 'loss', 'push'))
        / nullif(sum(stake_units) filter (where outcome in ('win', 'loss', 'push')), 0) as roi,
      avg(expected_value_per_unit) as average_ev,
      min(decision_at) as first_decision_at, max(decision_at) as last_decision_at
    from nhl.recommendations
    where status = 'recommended'
      and (p_season is null or season = p_season)
      and (p_market = 'all' or market = p_market)
      and (p_from is null or decision_at >= p_from)
    group by grouping sets ((policy_version, market), (policy_version))
  )
  select jsonb_build_object(
    'metrics', (select coalesce(jsonb_agg(to_jsonb(m)), '[]'::jsonb) from nhl.recommendation_summary(p_season,p_market,p_from) m),
    'seasons', (select coalesce(jsonb_agg(season order by season desc), '[]'::jsonb) from (select distinct season from nhl.recommendations) s),
    'by_day', (select coalesce(jsonb_agg(to_jsonb(d) order by d.date, d.market), '[]'::jsonb) from by_day d),
    'roi_error', (select coalesce(jsonb_agg(to_jsonb(e) order by e.market), '[]'::jsonb) from roi_error e),
    'closing', (select coalesce(jsonb_agg(to_jsonb(c) order by c.market), '[]'::jsonb) from closing c),
    'by_price', (select coalesce(jsonb_agg(to_jsonb(b) order by b.rule, b.ord), '[]'::jsonb) from by_price b),
    'by_rule', (select coalesce(jsonb_agg(to_jsonb(p) order by p.first_decision_at, p.market), '[]'::jsonb) from by_rule p)
  );
$$;

notify pgrst, 'reload schema';
commit;
