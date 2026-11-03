-- Retry the morning nhl-daily dispatch when pg_cron missed it.
-- On 2026-11-03 every pg_cron job in the shared project failed with
-- "job startup timeout" from 12:15 to 16:42 UTC, so the 16:15 dispatch never
-- reached GitHub. The named schedule is updated in place, so rerunning creates
-- no duplicate.
-- It fires at 16:49 and 17:49 UTC only when neither a dispatch has been queued
-- today nor ratings for today's Pacific site day are published (a manual run
-- counts), so a normal day sends nothing. Minute 49 avoids the live
-- dispatchers' offsets 0 to 3 (mod 5).
begin;
do $$
begin
  if not exists (select 1 from vault.secrets where name = 'github_dispatch_pat') then
    raise exception 'Missing github_dispatch_pat in Vault';
  end if;
end $$;

select cron.schedule('nhl-daily-dispatch-retry', '49 16,17 * 10-12,1-4 *', $dispatch$
  select net.http_post(
    url := 'https://api.github.com/repos/renenunezg/momentumnhl/dispatches',
    headers := jsonb_build_object(
      'Authorization', 'Bearer ' || (select decrypted_secret
        from vault.decrypted_secrets where name = 'github_dispatch_pat'),
      'Accept', 'application/vnd.github+json',
      'User-Agent', 'supabase-pg-cron',
      'X-GitHub-Api-Version', '2022-11-28'
    ),
    body := jsonb_build_object('event_type', 'nhl-daily'),
    timeout_milliseconds := 15000
  ) where not exists (
    select 1 from cron.job_run_details as run
    join cron.job using (jobid)
    where cron.job.jobname in ('nhl-daily-dispatch', 'nhl-daily-dispatch-retry')
      and run.status = 'succeeded'
      and run.return_message = '1 row'
      and run.start_time >= date_trunc('day', now())
  ) and not exists (
    select 1 from nhl.team_ratings
    where as_of = (now() at time zone 'America/Los_Angeles')::date
  );
$dispatch$);
commit;
