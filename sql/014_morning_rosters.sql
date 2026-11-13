-- Each club's roster as the NHL API listed it before the game, kept to learn
-- how much of tonight's lineup a morning run can know. The roster names who
-- is with the club, not who dresses and not who is hurt. Research only: no
-- forecast or decision reads it and the site does not show it, so there is
-- no anon access and no site_revalidate trigger. One row per team and game,
-- the first one fetched; a row fetched at or after the start is refused.
begin;

create table nhl.morning_rosters (
  game_id text not null,
  team_abbr text not null,
  game_date date not null,
  start_date timestamptz not null,
  fetched_at timestamptz not null,
  skater_ids int[] not null,
  goalie_ids int[] not null,
  primary key (game_id, team_abbr),
  constraint morning_roster_pregame check (fetched_at < start_date)
);

alter table nhl.morning_rosters enable row level security;
grant select, insert on nhl.morning_rosters to service_role;

notify pgrst, 'reload schema';
commit;
