-- The record by pick rule. recommendation_dashboard adds by_rule: every
-- recommended prediction grouped by the policy_version that made it, per
-- market and pooled, with its record, ROI, and average expected value, so the
-- site can show the current rule's expectation apart from retired rules'.
-- Every earlier field is unchanged.
begin;

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
    'by_rule', (select coalesce(jsonb_agg(to_jsonb(p) order by p.first_decision_at, p.market), '[]'::jsonb) from by_rule p)
  );
$$;

notify pgrst, 'reload schema';
commit;
