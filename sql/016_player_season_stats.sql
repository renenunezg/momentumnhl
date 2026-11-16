-- One raw season stat line per graded player for the player grade cards,
-- in the shape nfl.player_season_stats has. `stats` holds the values in the
-- key order and scale backend/players/season_stats.py (SKATER_LINE,
-- GOALIE_LINE) defines; momentumweb decodes it. The players run replaces the
-- season's rows in the transaction that replaces nhl.position_traits. No
-- forecast or decision reads it.
begin;

create table nhl.player_season_stats (
  season smallint not null,
  athlete_id text not null,
  position_group text not null check (position_group in ('F', 'D', 'G')),
  week smallint not null,
  stats integer[] not null,
  updated_at timestamptz not null default now(),
  primary key (season, athlete_id)
);

alter table nhl.player_season_stats enable row level security;
create policy "anon read" on nhl.player_season_stats for select to anon, authenticated using (true);
grant select on nhl.player_season_stats to anon, authenticated;
grant select, insert, delete on nhl.player_season_stats to service_role;

create trigger site_revalidate after insert or update or delete on nhl.player_season_stats
  for each statement execute function public.site_revalidate();

notify pgrst, 'reload schema';
commit;
