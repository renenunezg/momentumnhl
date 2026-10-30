-- One frozen close per game. The live worker holds the latest valid pregame
-- partner quote in memory and inserts it once when the game starts; the
-- database then refuses every change. The 15-minute closing window lives in
-- backend/config.py (CLOSING_LINE_MINUTES), where the selection happens.
-- The close is a market benchmark only: it grades nothing and no forecast
-- reads one. A game with no row has no close, and none is backfilled.
begin;

create table nhl.closing_lines (
  game_id text primary key,
  season int not null,
  start_date timestamptz not null,
  source text not null,
  -- The provider's own stamp on the quote, and when the worker fetched it.
  observed_at timestamptz not null,
  fetched_at timestamptz not null,
  -- The shared cross-sport shape: the home line, negative when home is
  -- favored, with its two prices. In the NHL the spread is the puck line.
  home_spread float8,
  spread_home_price float8,
  spread_away_price float8,
  total float8,
  over_price float8,
  under_price float8,
  home_moneyline float8,
  away_moneyline float8,
  constraint closing_line_pregame check (observed_at <= fetched_at and fetched_at < start_date),
  constraint closing_line_prices check (
    (home_moneyline is null or
     (abs(home_moneyline) >= 100 and abs(home_moneyline) < 'Infinity'::float8)) and
    (away_moneyline is null or
     (abs(away_moneyline) >= 100 and abs(away_moneyline) < 'Infinity'::float8)) and
    (total is null or (total > 0 and total < 'Infinity'::float8)) and
    (home_spread is null or abs(home_spread) < 'Infinity'::float8)
  )
);

-- Inserted once, pregame, and never changed. A row fetched at or after the
-- scheduled start is an in-play line, and one arriving more than ten minutes
-- after the start is a stale worker's; both are refused.
create function nhl.freeze_closing_line() returns trigger
language plpgsql as $$
begin
  if tg_op <> 'INSERT' then
    raise exception 'nhl.closing_lines is frozen: % is not allowed', tg_op;
  end if;
  if new.fetched_at >= new.start_date then
    raise exception 'closing line for % was fetched at or after the scheduled start', new.game_id;
  end if;
  if now() > new.start_date + interval '10 minutes' then
    raise exception 'closing line for % arrived more than 10 minutes after the start', new.game_id;
  end if;
  return new;
end;
$$;
create trigger freeze_closing_lines
before insert or update or delete on nhl.closing_lines
for each row execute function nhl.freeze_closing_line();
create trigger no_truncate_closing_lines
before truncate on nhl.closing_lines
for each statement execute function nhl.freeze_closing_line();

alter table nhl.closing_lines enable row level security;
create policy "anon read" on nhl.closing_lines for select to anon, authenticated using (true);
grant select on nhl.closing_lines to anon, authenticated;
grant select, insert on nhl.closing_lines to service_role;
-- No site_revalidate trigger: the site shows a close only beside a graded
-- result, and grading already revalidates.

-- The morning pipeline still archives its decision-time quotes here, but the
-- live worker no longer writes closing stamps to it, and nothing the site
-- shows pregame reads it, so an insert must not revalidate the site.
drop trigger site_revalidate on nhl.market_snapshots;

-- The close is read from its own row instead of being reselected from the
-- snapshot archive on every read. Column names and positions are unchanged.
create or replace view nhl.live_predictions with (security_invoker = true) as
select r.game_id, r.season, r.game_date, r.start_date, r.home_team, r.away_team,
  r.home_goals, r.away_goals, r.last_period_type,
  p.home_lambda, p.away_lambda, p.home_win_prob, p.model_total,
  p.as_of as forecast_as_of, p.recorded_at as forecast_recorded_at, p.model_version,
  r.source, r.source_fetched_at,
  coalesce(p.pure_home_win_prob, p.home_win_prob) as pure_home_win_prob,
  coalesce(p.pure_model_total, p.model_total) as pure_model_total,
  c.home_moneyline as closing_home_price, c.away_moneyline as closing_away_price,
  c.total as closing_total,
  c.source as closing_source,
  c.observed_at as closing_observed_at
from nhl.game_results r
left join lateral (
  select f.* from nhl.forecast_snapshots f
  where f.game_id = r.game_id and f.season = r.season
    and f.home_team_abbr = r.home_team_abbr
    and f.away_team_abbr = r.away_team_abbr
    and f.recorded_at < r.start_date and f.as_of < r.start_date
  order by f.recorded_at desc, f.snapshot_id desc limit 1
) p on true
left join nhl.closing_lines c on c.game_id = r.game_id
where r.start_date < now();

notify pgrst, 'reload schema';
commit;
