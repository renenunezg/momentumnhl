-- The partner feed's three-way line on the result after regulation, kept
-- beside the morning quote it arrived with. Its draw price is the market's
-- own chance of overtime. Research only: no forecast or decision reads it.
begin;
alter table nhl.market_snapshots
  add column home_regulation_price float8,
  add column draw_price float8,
  add column away_regulation_price float8;
notify pgrst, 'reload schema';
commit;
