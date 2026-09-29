-- An independently observed public listing can corroborate an exact partner
-- offer. It never replaces the original provider timestamp or price.
begin;

create function nhl.quote_verification_valid(
  evidence jsonb, expected_game text, expected_start timestamptz,
  expected_market text, expected_side text, expected_point float8,
  expected_price float8, expected_provider text, fetched timestamptz,
  decision timestamptz, published timestamptz, receipt jsonb
) returns boolean language plpgsql stable set search_path = pg_catalog as $$
declare
  observed timestamptz;
  response_date timestamptz;
  source_start timestamptz;
begin
  observed := (evidence->>'observed_at')::timestamptz;
  response_date := (evidence->>'response_date')::timestamptz;
  source_start := (evidence->>'source_start_date')::timestamptz;
  return coalesce(
    evidence->>'method' = 'draftkings_public_listing_v1' and
    expected_provider = 'draftkings' and
    evidence->>'provider_key' = expected_provider and
    evidence->>'game_id' = expected_game and
    (evidence->>'start_date')::timestamptz = expected_start and
    source_start between expected_start and expected_start + interval '15 minutes' and
    evidence->>'market' = expected_market and
    evidence->>'side' = expected_side and
    evidence ? 'point' and
    (evidence->>'point')::float8 is not distinct from expected_point and
    (evidence->>'price')::float8 = expected_price and
    evidence->>'source_url' = 'https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/?tb_edate=n7days&tb_eg=42133&tb_emt=0' and
    evidence->>'sha256' ~ '^[0-9a-f]{64}$' and
    evidence->>'source_event_id' ~ '^[0-9]+$' and
    ((expected_market = 'h2h' and
      evidence->>'source_market_id' ~ '^0ML[0-9]+$' and
      evidence->>'source_selection_id' = (evidence->>'source_market_id') ||
        case expected_side when 'home' then '_1' when 'away' then '_3' end) or
     (expected_market = 'totals' and
      evidence->>'source_market_id' ~ '^0OU[0-9]+$' and
      evidence->>'source_selection_id' = (evidence->>'source_market_id') ||
        case expected_side when 'over' then 'O' when 'under' then 'U' end ||
        round(expected_point * 100)::bigint::text ||
        case expected_side when 'over' then '_1' when 'under' then '_3' end)) and
    fetched <= observed and observed <= decision and decision <= published and
    published < expected_start and published - observed <= interval '15 minutes' and
    observed - response_date between interval '-5 seconds' and interval '300 seconds' and
    (evidence->>'response_age_seconds')::float8 between 0 and 300 and
    receipt->>'sha256' = evidence->>'sha256' and
    (receipt->>'observed_at')::timestamptz = observed,
    false
  );
exception when invalid_text_representation or datetime_field_overflow
  or invalid_datetime_format or numeric_value_out_of_range then
  return false;
end;
$$;

alter table nhl.recommendations drop constraint recommendation_eligibility_v1;
alter table nhl.recommendations add constraint recommendation_eligibility_v1_1 check (
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
   ((market = 'h2h' and probability_edge >= 0.13 and expected_value_per_unit > 0) or
    (market = 'totals' and edge_points >= 1 and edge_points < 'Infinity'::float8
     and market_total = point))) is true
);

commit;
