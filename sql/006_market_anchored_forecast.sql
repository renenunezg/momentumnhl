-- A game the books quote publishes a market-anchored forecast: its goal
-- rates, win probability, total, and fair prices carry the blend, beside the
-- pure model's probability and total and the market consensus for each. Rows
-- published before this migration are pure and leave the four columns null.
-- nhl-picks-v2 prices both markets from that published forecast. A moneyline
-- needs 4.5 points of edge over break-even instead of the sheet's 13, and a
-- total half a goal on the blended total, which is the sheet's one goal on
-- the pure total. Earlier picks stay valid and keep their policy version.
begin;

-- Both tables take the columns in the same order: the archive trigger copies
-- a projection row into its snapshot by position.
alter table nhl.game_projections
  add column pure_home_win_prob float8, add column pure_model_total float8,
  add column market_home_prob float8, add column market_total float8;
alter table nhl.forecast_snapshots
  add column pure_home_win_prob float8, add column pure_model_total float8,
  add column market_home_prob float8, add column market_total float8;

alter table nhl.recommendations drop constraint recommendation_eligibility_v1_1;
alter table nhl.recommendations add constraint recommendation_eligibility_v2 check (
  (status = 'no_play' and stake_units = 0) or
  (status = 'recommended' and stake_units = 1 and missing_input_count = 0 and
   selection is not null and side is not null and
   ((market = 'h2h' and side in ('home', 'away') and point is null) or
    (market = 'totals' and side in ('over', 'under') and point is not null
     and point > 0 and point < 'Infinity'::float8 and point * 2 = round(point * 2))) and
   selection = case side when 'home' then home_team when 'away' then away_team
                         when 'over' then 'Over' else 'Under' end and
   abs(price) >= 100 and abs(price) < 'Infinity'::float8 and
   provider is not null and provider_key is not null and
   provider_event_id = game_id and provider_start_date = start_date and
   provider_last_update <= market_fetched_at and market_fetched_at <= decision_at and
   (published_at - provider_last_update <= interval '24 hours' or
    nhl.quote_verification_valid(
      data_flags->'quote_verification', game_id, start_date, market, side, point,
      price, provider_key, market_fetched_at, decision_at, published_at,
      source_timestamps->'quote_verification')) and
   published_at - forecast_as_of <= interval '24 hours' and
   win_probability > 0 and win_probability < 1 and
   push_probability >= 0 and win_probability + push_probability <= 1 and
   expected_value_per_unit > -1 and expected_value_per_unit < 'Infinity'::float8 and
   ((market = 'h2h' and probability_edge >= 0.045 and expected_value_per_unit > 0) or
    (market = 'totals' and edge_points >= 0.5 and edge_points < 'Infinity'::float8
     and market_total = point))) is true
);

create or replace view nhl.recommendation_performance with (security_invoker = true) as
  with segments as (
    select r.*, s.kind, s.label
    from nhl.recommendations r
    cross join lateral (
      select * from (values
        ('overall', 'All picks'), ('market', r.market),
        ('month', to_char(r.decision_at at time zone 'America/Los_Angeles', 'YYYY-MM')),
        ('policy', r.policy_version),
        ('edge', case when r.market = 'h2h' then
            case when r.probability_edge < 0.08 then '4.5-8 pp'
                 when r.probability_edge < 0.13 then '8-13 pp'
                 when r.probability_edge < 0.16 then '13-16 pp'
                 when r.probability_edge < 0.20 then '16-20 pp' else '20+ pp' end
          else case when r.edge_points < 0.75 then '0.5-0.75 goals'
                    when r.edge_points < 1 then '0.75-1 goals'
                    when r.edge_points < 1.5 then '1-1.5 goals'
                    when r.edge_points < 2 then '1.5-2 goals' else '2+ goals' end end)
      ) v(kind,label)
      union all
      select 'side', r.market || ':' || case
        when r.market = 'totals' then r.side
        else case when r.price < -100 then 'favorite' when r.price > 100 then 'underdog' else 'even' end
      end
      where r.status = 'recommended'
    ) s(kind,label)
  ), aggregates as (
    select null::integer as season, kind as segment_kind, label as segment,
      count(distinct game_id) filter (where status = 'recommended') as unique_games,
      count(*) filter (where status = 'recommended') as picks,
      count(*) filter (where status = 'no_play') as no_plays,
      count(*) filter (where status = 'recommended' and outcome = 'pending') as pending,
      count(*) filter (where outcome = 'win') as wins,
      count(*) filter (where outcome = 'loss') as losses,
      count(*) filter (where outcome = 'push') as pushes,
      count(*) filter (where status = 'recommended' and outcome = 'void') as voids,
      coalesce(sum(stake_units) filter (where outcome in ('win','loss','push')), 0) as staked_units,
      coalesce(sum(profit_units) filter (where outcome in ('win','loss','push')), 0) as profit_units,
      avg(expected_value_per_unit) filter (where status = 'recommended') as average_ev,
      min(decision_at) as first_decision_at,
      max(decision_at) as last_decision_at,
      max(graded_at) as last_graded_at
    from segments group by kind, label
  )
  select *, profit_units / nullif(staked_units, 0) as roi,
    wins::float8 / nullif(wins + losses, 0) as win_rate,
    wins + losses + pushes < 30 as thin_sample
  from aggregates;

create or replace function nhl.recommendation_summary(
  p_season integer default null,
  p_market text default 'all',
  p_from timestamptz default null
) returns setof nhl.recommendation_performance
language sql stable security invoker set search_path = pg_catalog, nhl as $$
  with filtered as (
    select * from nhl.recommendations
    where (p_season is null or season = p_season)
      and (p_market = 'all' or market = p_market)
      and (p_from is null or decision_at >= p_from)
  ), segments as (
    select r.*, s.kind, s.label
    from filtered r
    cross join lateral (
      select * from (values
        ('overall', 'All picks'), ('market', r.market),
        ('month', to_char(r.decision_at at time zone 'America/Los_Angeles', 'YYYY-MM')),
        ('policy', r.policy_version),
        ('edge', case when r.market = 'h2h' then
            case when r.probability_edge < 0.08 then '4.5-8 pp'
                 when r.probability_edge < 0.13 then '8-13 pp'
                 when r.probability_edge < 0.16 then '13-16 pp'
                 when r.probability_edge < 0.20 then '16-20 pp' else '20+ pp' end
          else case when r.edge_points < 0.75 then '0.5-0.75 goals'
                    when r.edge_points < 1 then '0.75-1 goals'
                    when r.edge_points < 1.5 then '1-1.5 goals'
                    when r.edge_points < 2 then '1.5-2 goals' else '2+ goals' end end)
      ) v(kind,label)
      union all
      select 'side', r.market || ':' || case
        when r.market = 'totals' then r.side
        else case when r.price < -100 then 'favorite' when r.price > 100 then 'underdog' else 'even' end
      end
      where r.status = 'recommended'
    ) s(kind,label)
  ), aggregates as (
    select p_season as season, kind as segment_kind, label as segment,
      count(distinct game_id) filter (where status = 'recommended') as unique_games,
      count(*) filter (where status = 'recommended') as picks,
      count(*) filter (where status = 'no_play') as no_plays,
      count(*) filter (where status = 'recommended' and outcome = 'pending') as pending,
      count(*) filter (where outcome = 'win') as wins,
      count(*) filter (where outcome = 'loss') as losses,
      count(*) filter (where outcome = 'push') as pushes,
      count(*) filter (where status = 'recommended' and outcome = 'void') as voids,
      coalesce(sum(stake_units) filter (where outcome in ('win','loss','push')), 0) as staked_units,
      coalesce(sum(profit_units) filter (where outcome in ('win','loss','push')), 0) as profit_units,
      avg(expected_value_per_unit) filter (where status = 'recommended') as average_ev,
      min(decision_at) as first_decision_at,
      max(decision_at) as last_decision_at,
      max(graded_at) as last_graded_at
    from segments group by kind, label
  )
  select *, profit_units / nullif(staked_units, 0) as roi,
    wins::float8 / nullif(wins + losses, 0) as win_rate,
    wins + losses + pushes < 30 as thin_sample
  from aggregates;
$$;

notify pgrst, 'reload schema';
commit;
