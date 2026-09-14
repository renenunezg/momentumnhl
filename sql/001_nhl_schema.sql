-- nhl schema v1: the sole interface between momentumnhl and momentumweb.
-- Applied to the shared momentum Supabase project. RLS with anon read on
-- every site-read table, mirroring the nfl schema's policy shape.

begin;

create schema if not exists nhl;

create table nhl.teams (
  team_abbr text primary key,
  team text not null,
  conference text,
  division text,
  color text,
  logo_light text,
  logo_dark text
);

-- One row per team per model day. The window columns are the sheet's
-- Expected Goals blocks; the strengths and rating are its Model Outputs.
create table nhl.team_ratings (
  as_of date not null,
  team_abbr text not null,
  team text not null,
  model_version text not null,
  window_games_home int not null,
  window_games_away int not null,
  home_xgf float8 not null,
  home_xga float8 not null,
  away_xgf float8 not null,
  away_xga float8 not null,
  home_attack float8 not null,
  home_defense float8 not null,
  away_attack float8 not null,
  away_defense float8 not null,
  rating float8 not null,
  insufficient_window bool not null,
  published_at timestamptz not null default clock_timestamp(),
  primary key (as_of, team_abbr)
);

create table nhl.game_projections (
  game_id text primary key,
  season int not null,
  game_date date not null,
  start_date timestamptz not null,
  as_of timestamptz not null,
  model_version text not null,
  home_team_abbr text not null,
  away_team_abbr text not null,
  home_team text not null,
  away_team text not null,
  home_lambda float8 not null,
  away_lambda float8 not null,
  home_win_prob float8 not null,
  away_win_prob float8 not null,
  model_total float8 not null,
  home_fair_decimal float8 not null,
  away_fair_decimal float8 not null,
  home_fair_price float8 not null,
  away_fair_price float8 not null,
  home_minimum_price float8 not null,
  away_minimum_price float8 not null,
  missing_input_count int not null
);

-- Only forecasts actually published before puck drop qualify for live
-- grading. Database receipt time is authoritative, never a caller's as_of.
create table nhl.forecast_snapshots (
  snapshot_id bigint generated always as identity primary key,
  recorded_at timestamptz not null default clock_timestamp(),
  like nhl.game_projections including defaults
);
create index forecast_snapshots_game_time
  on nhl.forecast_snapshots (game_id, recorded_at desc, snapshot_id desc);

create function nhl.guard_pregame_projection() returns trigger
language plpgsql set search_path = pg_catalog, nhl as $$
begin
  if TG_OP = 'DELETE' then
    raise exception 'Game projections cannot be deleted; publish with an upsert';
  end if;
  if new.as_of > clock_timestamp() or new.as_of >= new.start_date
      or new.start_date <= clock_timestamp() then
    return null;
  end if;
  if TG_OP = 'UPDATE' then
    if old.start_date <= clock_timestamp() or new.as_of < old.as_of then
      return null;
    end if;
    if new.game_id <> old.game_id or new.season <> old.season
        or new.home_team_abbr <> old.home_team_abbr
        or new.away_team_abbr <> old.away_team_abbr then
      raise exception 'Forecast game identity cannot change';
    end if;
    if new is not distinct from old then
      return null;
    end if;
  end if;
  return new;
end;
$$;
create trigger guard_pregame_projection
before insert or update or delete on nhl.game_projections
for each row execute function nhl.guard_pregame_projection();

create function nhl.archive_pregame_projection() returns trigger
language plpgsql set search_path = pg_catalog, nhl as $$
begin
  insert into nhl.forecast_snapshots
    overriding system value
    select nextval(pg_get_serial_sequence('nhl.forecast_snapshots', 'snapshot_id')),
      clock_timestamp(), new.*;
  return new;
end;
$$;
create trigger archive_pregame_projection
after insert or update on nhl.game_projections
for each row execute function nhl.archive_pregame_projection();

create function nhl.check_forecast_commit_cutoff() returns trigger
language plpgsql set search_path = pg_catalog, nhl as $$
begin
  if clock_timestamp() >= new.start_date or new.recorded_at >= new.start_date
      or new.as_of >= new.start_date then
    raise exception 'Forecast publication crossed puck drop; retry to skip it';
  end if;
  return new;
end;
$$;
create constraint trigger forecast_commit_cutoff
after insert on nhl.forecast_snapshots
deferrable initially deferred
for each row execute function nhl.check_forecast_commit_cutoff();

create function nhl.reject_forecast_mutation() returns trigger
language plpgsql set search_path = pg_catalog, nhl as $$
begin
  raise exception 'Published forecast snapshots are immutable';
end;
$$;
create trigger immutable_forecast_snapshots
before update or delete or truncate on nhl.forecast_snapshots
for each statement execute function nhl.reject_forecast_mutation();
create trigger no_truncate_game_projections
before truncate on nhl.game_projections
for each statement execute function nhl.reject_forecast_mutation();

-- Every partner feed read, so a decision's price can be audited.
create table nhl.market_snapshots (
  game_id text not null,
  provider_key text not null,
  fetched_at timestamptz not null,
  provider_last_update timestamptz,
  home_price float8,
  away_price float8,
  total_line float8,
  over_price float8,
  under_price float8,
  puck_line float8,
  home_puck_price float8,
  away_puck_price float8,
  primary key (game_id, provider_key, fetched_at)
);

create table nhl.game_results (
  game_id text primary key,
  season int not null,
  game_date date not null,
  start_date timestamptz not null,
  home_team_abbr text not null,
  away_team_abbr text not null,
  home_team text not null,
  away_team text not null,
  home_goals int not null check (home_goals >= 0),
  away_goals int not null check (away_goals >= 0),
  last_period_type text not null check (last_period_type in ('REG', 'OT', 'SO')),
  source text not null,
  source_fetched_at timestamptz not null,
  check (home_goals <> away_goals)
);

-- Results joined to the last forecast published before puck drop. Missing
-- forecasts stay visible so coverage is honest.
create view nhl.live_predictions with (security_invoker = true) as
select r.game_id, r.season, r.game_date, r.start_date, r.home_team, r.away_team,
  r.home_goals, r.away_goals, r.last_period_type,
  p.home_lambda, p.away_lambda, p.home_win_prob, p.model_total,
  p.as_of as forecast_as_of, p.recorded_at as forecast_recorded_at, p.model_version,
  r.source, r.source_fetched_at
from nhl.game_results r
left join lateral (
  select f.* from nhl.forecast_snapshots f
  where f.game_id = r.game_id and f.season = r.season
    and f.home_team_abbr = r.home_team_abbr
    and f.away_team_abbr = r.away_team_abbr
    and f.recorded_at < r.start_date and f.as_of < r.start_date
  order by f.recorded_at desc, f.snapshot_id desc limit 1
) p on true
where r.start_date < now();

create table nhl.backtest_predictions (
  game_id text primary key,
  season int not null,
  game_date date not null,
  home_team text not null,
  away_team text not null,
  home_goals int not null,
  away_goals int not null,
  last_period_type text,
  home_lambda float8 not null,
  away_lambda float8 not null,
  home_win_prob float8 not null,
  model_total float8 not null
);

-- Probability and total accuracy for the live record or the backtest.
create function nhl.forecast_accuracy(p_source text default 'live') returns jsonb
language sql stable security invoker set search_path = pg_catalog, nhl as $$
  with selected as materialized (
    select season, home_win_prob, model_total, home_goals, away_goals
    from nhl.live_predictions where p_source = 'live' and home_win_prob is not null
    union all
    select season, home_win_prob, model_total, home_goals, away_goals
    from nhl.backtest_predictions where p_source = 'backtest'
  ), scored as (
    select season,
      least(greatest(home_win_prob, 1e-6), 1 - 1e-6) as p,
      (home_goals > away_goals)::int as home_won,
      model_total, (home_goals + away_goals)::float8 as total
    from selected
  ), by_season as (
    select season, count(*) as n,
      -avg(home_won * ln(p) + (1 - home_won) * ln(1 - p)) as log_loss,
      avg((p - home_won) ^ 2) as brier,
      avg(((p > 0.5) = (home_won = 1))::int) as accuracy,
      avg(home_won) as home_win_rate,
      avg(abs(model_total - total)) as total_mae,
      avg(model_total - total) as total_bias
    from scored group by season
  ), overall as (
    select count(*) as n,
      -avg(home_won * ln(p) + (1 - home_won) * ln(1 - p)) as log_loss,
      avg((p - home_won) ^ 2) as brier,
      avg(((p > 0.5) = (home_won = 1))::int) as accuracy,
      avg(home_won) as home_win_rate,
      avg(abs(model_total - total)) as total_mae,
      avg(model_total - total) as total_bias
    from scored
  ), calibration as (
    select least(floor(p * 10), 9)::int as bin, count(*) as n,
      avg(p) as predicted, avg(home_won) as observed
    from scored group by 1 order by 1
  )
  select jsonb_build_object(
    'overall', case when (select n from overall) > 0
      then (select to_jsonb(o) from overall o) else null end,
    'by_season', (select coalesce(jsonb_agg(to_jsonb(s) order by s.season desc), '[]'::jsonb) from by_season s),
    'calibration', (select coalesce(jsonb_agg(to_jsonb(c) order by c.bin), '[]'::jsonb) from calibration c),
    'missing_forecast', (select count(*) from nhl.live_predictions
      where p_source = 'live' and home_win_prob is null)
  );
$$;

-- Fixture observations the ledger settles against.
create table nhl.recommendation_schedule (
  game_id text primary key,
  season int not null,
  start_date timestamptz,
  home_team text not null,
  away_team text not null,
  game_status text not null,
  completed bool not null,
  home_goals int,
  away_goals int,
  observed_at timestamptz not null check (observed_at <= clock_timestamp()),
  check (not completed or (home_goals >= 0 and away_goals >= 0 and start_date < observed_at) is true)
);

-- One decision per game and market at the sheet's thresholds. Moneylines
-- need 13 points of edge over the break-even probability; totals need a
-- full goal between the model total and the posted line.
create table nhl.recommendations (
  game_id text not null,
  market text not null check (market in ('h2h', 'totals')),
  season int not null,
  game_date date not null,
  start_date timestamptz not null,
  home_team text not null,
  away_team text not null,
  model_version text not null,
  forecast_as_of timestamptz not null,
  missing_input_count int not null,
  policy_version text not null,
  decision_at timestamptz not null,
  published_at timestamptz not null default clock_timestamp(),
  status text not null check (status in ('recommended', 'no_play')),
  reason text not null,
  selection text,
  side text check (side in ('home', 'away', 'over', 'under')),
  point float8,
  price float8,
  provider text,
  provider_key text,
  market_fetched_at timestamptz,
  provider_event_id text,
  provider_start_date timestamptz,
  provider_last_update timestamptz,
  win_probability float8,
  push_probability float8,
  probability_edge float8,
  edge_points float8,
  expected_value_per_unit float8,
  stake_units float8 not null,
  kelly_fraction float8,
  minimum_price float8,
  home_lambda float8 not null,
  away_lambda float8 not null,
  model_total float8 not null,
  market_total float8,
  source_timestamps jsonb not null,
  data_flags jsonb not null,
  pricing_weights jsonb not null,
  outcome text not null default 'pending'
    check (outcome in ('pending', 'win', 'loss', 'push', 'void', 'no_play')),
  home_goals int,
  away_goals int,
  profit_units float8,
  graded_at timestamptz,
  settlement_reason text,
  result_source_at timestamptz,
  primary key (game_id, market),
  check (forecast_as_of <= decision_at and decision_at <= published_at),
  check (published_at < start_date),
  check (home_lambda > 0 and home_lambda < 'Infinity'::float8
     and away_lambda > 0 and away_lambda < 'Infinity'::float8
     and model_total > 0 and model_total < 'Infinity'::float8),
  constraint recommendation_eligibility_v1 check (
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
     published_at - provider_last_update <= interval '24 hours' and
     published_at - forecast_as_of <= interval '24 hours' and
     win_probability > 0 and win_probability < 1 and
     push_probability >= 0 and win_probability + push_probability <= 1 and
     expected_value_per_unit > -1 and expected_value_per_unit < 'Infinity'::float8 and
     ((market = 'h2h' and probability_edge >= 0.13 and expected_value_per_unit > 0) or
      (market = 'totals' and edge_points >= 1 and edge_points < 'Infinity'::float8
       and market_total = point))) is true),
  check ((outcome = 'pending' and graded_at is null and profit_units is null) or
    (outcome <> 'pending' and graded_at is not null and profit_units is not null))
);

create function nhl.protect_recommendation() returns trigger
language plpgsql set search_path = pg_catalog, nhl as $$
declare
  balance float8;
  expected_outcome text;
  expected_profit float8;
  source_name text;
  observed timestamptz;
  result nhl.recommendation_schedule;
begin
  if TG_OP = 'DELETE' then
    raise exception 'Recommendation history cannot be deleted';
  end if;
  if TG_OP = 'INSERT' then
    -- Caller-supplied historical timestamps must never admit a past pick.
    new.published_at := clock_timestamp();
    if exists (select 1 from nhl.game_results where game_id = new.game_id)
       or exists (select 1 from nhl.forecast_snapshots
                  where game_id = new.game_id and start_date <= clock_timestamp()) then
      raise exception 'Cannot insert a decision for a previously started game';
    end if;
    if new.outcome <> 'pending' then
      raise exception 'New recommendations must be pending';
    end if;
  end if;
  if TG_OP = 'UPDATE' then
    if old.outcome <> 'pending' and new is distinct from old then
      raise exception 'Settled recommendations are immutable';
    end if;
    if old.status = 'recommended' or old.start_date <= clock_timestamp() then
      if (to_jsonb(new) - array['outcome','home_goals','away_goals','profit_units','graded_at','settlement_reason','result_source_at'])
        is distinct from
         (to_jsonb(old) - array['outcome','home_goals','away_goals','profit_units','graded_at','settlement_reason','result_source_at']) then
        raise exception 'Published picks and started game decisions are frozen';
      end if;
    end if;
    if old.status = 'no_play' and old.outcome = 'pending' and new.outcome = 'pending'
       and new is distinct from old then
      new.published_at := clock_timestamp();
    end if;
  end if;
  if new.outcome = 'pending' then
    if new.home_goals is not null or new.away_goals is not null then
      raise exception 'Pending recommendations cannot contain final scores';
    end if;
  else
    if not (new.graded_at >= new.published_at and new.graded_at <= clock_timestamp()) then
      raise exception 'Invalid settlement timestamp';
    end if;
    select * into result from nhl.recommendation_schedule where game_id = new.game_id;
    if not (result.observed_at = new.result_source_at and result.observed_at <= new.graded_at
       and result.home_team = new.home_team and result.away_team = new.away_team) is true then
      raise exception 'Settlement requires a matching confirmed schedule observation';
    end if;
    if new.outcome = 'void' then
      if not (new.settlement_reason = 'schedule_change' and
          (result.start_date <> new.start_date
           or result.game_status in ('canceled', 'cancelled', 'postponed', 'ppd'))) is true then
        raise exception 'Void requires a schedule change';
      end if;
      expected_profit := 0;
    elsif new.outcome = 'no_play' and new.status = 'no_play' then
      if new.settlement_reason = 'confirmed_final' and new.graded_at < new.start_date then
        raise exception 'Cannot settle before puck drop';
      end if;
      expected_profit := 0;
    elsif new.status = 'recommended' and new.outcome in ('win', 'loss', 'push') then
      if not (new.graded_at >= new.start_date and new.home_goals >= 0
              and new.away_goals >= 0) is true then
        raise exception 'Settlement requires a started game and final scores';
      end if;
      if not (result.completed and result.start_date = new.start_date
         and result.home_goals = new.home_goals and result.away_goals = new.away_goals
         and new.settlement_reason = 'confirmed_final') is true then
        raise exception 'Scores must match a confirmed final schedule observation';
      end if;
      balance := case when new.market = 'h2h' then
        (new.home_goals::float8 - new.away_goals) * case new.side when 'home' then 1 else -1 end
        else (new.home_goals::float8 + new.away_goals - new.point) * case new.side when 'over' then 1 else -1 end end;
      if new.market = 'h2h' and balance = 0 then
        raise exception 'NHL games cannot end tied';
      end if;
      expected_outcome := case when balance > 0 then 'win' when balance < 0 then 'loss' else 'push' end;
      if new.outcome <> expected_outcome then
        raise exception 'Outcome disagrees with the recorded line and final score';
      end if;
      expected_profit := case new.outcome when 'win' then
        case when new.price > 0 then new.price / 100 else 100 / abs(new.price) end
        when 'loss' then -1 else 0 end;
    else
      raise exception 'Outcome is incompatible with the recorded decision';
    end if;
    if not (abs(new.profit_units - expected_profit) < 1e-10) is true then
      raise exception 'Profit disagrees with the recorded price and outcome';
    end if;
  end if;
  if new.outcome = 'pending' and new.status = 'recommended' then
    foreach source_name in array array['moneypuck', 'schedule', 'odds'] loop
      observed := (new.source_timestamps -> source_name ->> 'observed_at')::timestamptz;
      if not (observed <= new.decision_at and
          new.source_timestamps -> source_name ->> 'sha256' is not null) is true then
        raise exception 'Missing or future source receipt: %', source_name;
      end if;
      if new.published_at - observed > interval '24 hours' then
        raise exception 'Stale source receipt: %', source_name;
      end if;
    end loop;
    expected_profit := case when new.price > 0 then new.price / 100 else 100 / abs(new.price) end;
    if not (abs(new.expected_value_per_unit - (new.win_probability * expected_profit
          - (1 - new.win_probability - new.push_probability))) < 1e-9
      and abs(new.probability_edge - (new.win_probability / (1 - new.push_probability)
          - 1 / (1 + expected_profit))) < 1e-9) is true then
      raise exception 'EV and edge must match recorded probability and price';
    end if;
  end if;
  return new;
end
$$;
create trigger protect_recommendation before insert or update or delete on nhl.recommendations
  for each row execute function nhl.protect_recommendation();
create function nhl.reject_recommendation_truncate() returns trigger language plpgsql as $$
begin raise exception 'Recommendation history cannot be truncated'; end $$;
create trigger reject_recommendation_truncate before truncate on nhl.recommendations
  for each statement execute function nhl.reject_recommendation_truncate();

-- A transaction that begins pregame but commits after puck drop did not
-- make its decision public in time.
create function nhl.check_recommendation_commit() returns trigger language plpgsql as $$
begin
  if new.outcome = 'pending' and clock_timestamp() >= new.start_date then
    raise exception 'Recommendation commit crossed puck drop';
  end if;
  return new;
end $$;
create constraint trigger recommendation_commit_cutoff
after insert or update on nhl.recommendations
deferrable initially deferred
for each row execute function nhl.check_recommendation_commit();

create view nhl.recommendation_performance with (security_invoker = true) as
  with segments as (
    select r.*, s.kind, s.label
    from nhl.recommendations r
    cross join lateral (
      select * from (values
        ('overall', 'All picks'), ('market', r.market),
        ('month', to_char(r.decision_at at time zone 'America/Los_Angeles', 'YYYY-MM')),
        ('policy', r.policy_version),
        ('edge', case when r.market = 'h2h' then
            case when r.probability_edge < 0.16 then '13-16 pp'
                 when r.probability_edge < 0.20 then '16-20 pp' else '20+ pp' end
          else case when r.edge_points < 1.5 then '1-1.5 goals'
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

create function nhl.recommendation_summary(
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
            case when r.probability_edge < 0.16 then '13-16 pp'
                 when r.probability_edge < 0.20 then '16-20 pp' else '20+ pp' end
          else case when r.edge_points < 1.5 then '1-1.5 goals'
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

create function nhl.recommendation_dashboard(
  p_season integer default null, p_market text default 'all', p_from timestamptz default null
) returns jsonb language sql stable security invoker set search_path = pg_catalog, nhl as $$
  select jsonb_build_object(
    'metrics', (select coalesce(jsonb_agg(to_jsonb(m)), '[]'::jsonb) from nhl.recommendation_summary(p_season,p_market,p_from) m),
    'seasons', (select coalesce(jsonb_agg(season order by season desc), '[]'::jsonb) from (select distinct season from nhl.recommendations) s)
  );
$$;

-- Access: anon reads everything the site renders; the pipeline role writes.
alter table nhl.teams enable row level security;
alter table nhl.team_ratings enable row level security;
alter table nhl.game_projections enable row level security;
alter table nhl.forecast_snapshots enable row level security;
alter table nhl.market_snapshots enable row level security;
alter table nhl.game_results enable row level security;
alter table nhl.backtest_predictions enable row level security;
alter table nhl.recommendation_schedule enable row level security;
alter table nhl.recommendations enable row level security;

create policy "anon read" on nhl.teams for select to anon, authenticated using (true);
create policy "anon read" on nhl.team_ratings for select to anon, authenticated using (true);
create policy "anon read" on nhl.game_projections for select to anon, authenticated using (true);
create policy "anon read" on nhl.forecast_snapshots for select to anon, authenticated using (true);
create policy "anon read" on nhl.market_snapshots for select to anon, authenticated using (true);
create policy "anon read" on nhl.game_results for select to anon, authenticated using (true);
create policy "anon read" on nhl.backtest_predictions for select to anon, authenticated using (true);
create policy "anon read" on nhl.recommendations for select to anon, authenticated using (true);

grant usage on schema nhl to anon, authenticated, service_role;
grant select on nhl.teams, nhl.team_ratings, nhl.game_projections, nhl.forecast_snapshots,
  nhl.market_snapshots, nhl.game_results, nhl.live_predictions, nhl.backtest_predictions,
  nhl.recommendations, nhl.recommendation_performance to anon, authenticated;
grant select, insert, update on nhl.recommendation_schedule, nhl.recommendations to service_role;
grant select on nhl.recommendation_performance to service_role;
revoke all on function nhl.recommendation_dashboard(integer,text,timestamptz) from public;
revoke all on function nhl.recommendation_summary(integer,text,timestamptz) from public;
revoke all on function nhl.forecast_accuracy(text) from public;
grant execute on function nhl.recommendation_dashboard(integer,text,timestamptz),
  nhl.recommendation_summary(integer,text,timestamptz),
  nhl.forecast_accuracy(text) to anon, authenticated, service_role;

-- The site caches reads for an hour; every pipeline write asks it to refresh.
create trigger site_revalidate after insert or update or delete on nhl.teams
  for each statement execute function public.site_revalidate();
create trigger site_revalidate after insert or update or delete on nhl.team_ratings
  for each statement execute function public.site_revalidate();
create trigger site_revalidate after insert or update or delete on nhl.game_projections
  for each statement execute function public.site_revalidate();
create trigger site_revalidate after insert or update or delete on nhl.market_snapshots
  for each statement execute function public.site_revalidate();
create trigger site_revalidate after insert or update or delete on nhl.game_results
  for each statement execute function public.site_revalidate();
create trigger site_revalidate after insert or update or delete on nhl.backtest_predictions
  for each statement execute function public.site_revalidate();
create trigger site_revalidate after insert or update or delete on nhl.recommendations
  for each statement execute function public.site_revalidate();

notify pgrst, 'reload schema';
commit;
