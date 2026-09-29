-- Preserve independently observed prices for every side, not only selections
-- that became picks. These are archived pregame observations, not live quotes.
begin;
alter table nhl.market_snapshots
  add column quote_verifications jsonb not null default '{}'::jsonb
  check (jsonb_typeof(quote_verifications) = 'object');
notify pgrst, 'reload schema';
commit;
