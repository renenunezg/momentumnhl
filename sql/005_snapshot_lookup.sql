BEGIN;
-- One bounded archive response per slate, including the forecast cutoff.
CREATE INDEX market_snapshots_forecast_lookup
 ON nhl.market_snapshots (game_id, provider_key, fetched_at DESC);
CREATE FUNCTION nhl.latest_market_snapshots(p_game_ids text[], p_cutoffs jsonb DEFAULT '{}')
RETURNS SETOF nhl.market_snapshots LANGUAGE sql STABLE SET search_path = '' AS $$
 SELECT DISTINCT ON (game_id, provider_key) s.*
 FROM nhl.market_snapshots s
 WHERE game_id = ANY(p_game_ids)
   AND fetched_at <= coalesce((p_cutoffs->>game_id)::timestamptz, statement_timestamp())
 ORDER BY game_id, provider_key, fetched_at DESC
$$;
GRANT EXECUTE ON FUNCTION nhl.latest_market_snapshots(text[], jsonb) TO anon, authenticated, service_role;
NOTIFY pgrst, 'reload schema';
COMMIT;
