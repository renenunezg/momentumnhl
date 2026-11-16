-- Skater and goalie grades for the site's players page, in the shape the NFL
-- and CFB player pages read: one row per player and trait, with a `grade` row
-- on the 20 to 80 scale (50 average, 10 points a standard deviation among
-- qualified players at the position). position_group is F, D or G. Skater
-- traits are goals above an average skater at the position per 82 games;
-- the goalie trait is goals saved per 60 minutes above expected. The table
-- keeps the latest snapshot of each season: every run replaces the season's
-- rows. No forecast or decision reads it.
begin;

create table nhl.position_traits (
  season int not null,
  week int not null,
  as_of timestamptz not null,
  model_version text not null,
  athlete_id text not null,
  athlete_name text not null,
  headshot_url text,
  team text not null,
  position_group text not null check (position_group in ('F', 'D', 'G')),
  games float8 not null,
  trait text not null,
  value float8 not null,
  percentile float8 not null,
  kind text,
  in_grade bool not null,
  uncertainty float8,
  reliability float8,
  points_above_replacement float8,
  primary key (season, athlete_id, trait)
);
create index position_traits_group_trait_idx
  on nhl.position_traits (season, position_group, trait);

alter table nhl.position_traits enable row level security;
create policy "anon read" on nhl.position_traits for select to anon, authenticated using (true);
grant select on nhl.position_traits to anon, authenticated;
grant select, insert, delete on nhl.position_traits to service_role;

create trigger site_revalidate after insert or update or delete on nhl.position_traits
  for each statement execute function public.site_revalidate();

notify pgrst, 'reload schema';
commit;
