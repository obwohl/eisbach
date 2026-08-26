# How the forecast gets scheduled

The pipeline runs three times a day from `.github/workflows/forecast.yml`, at 04:00,
12:00 and 20:00 UTC (in Munich roughly 05/06, 13/14 and 21/22, depending on daylight
saving). There are three ways to start it, and the order matters.

## 1. `schedule:` — the default

The workflow carries its own cron schedule. This is the primary trigger and it depends
on nothing outside GitHub: no third-party service, no token, no account that can quietly
expire.

GitHub's scheduled triggers are known to be late under load, sometimes by tens of
minutes, and can be skipped entirely when the platform is busy. For a thrice-daily water
temperature forecast that is an acceptable trade for having no external dependency.

Note that GitHub disables scheduled workflows in repositories with no activity for 60
days. If the forecast goes quiet, check that first.

## 2. `workflow_dispatch:` — manual

Actions → *Eisbach Forecast* → **Run workflow**.

Use this to test a change, or to catch up after a missed run. Its real value is that it
exists at all: the workflow previously had neither `schedule` nor `workflow_dispatch`, so
when the external trigger stopped firing there was no way to start it by hand, and no way
to diagnose it either.

## 3. `repository_dispatch:` — optional external trigger

Still supported, for a more punctual schedule than GitHub's own. A service such as
[cron-job.org](https://cron-job.org) POSTs to the GitHub API:

- **URL:** `https://api.github.com/repos/obwohl/eisbach/dispatches`
- **Method:** `POST`
- **Headers:** `Accept: application/vnd.github+json`, `Authorization: Bearer <PAT>`
- **Body:** `{"event_type": "trigger_forecast"}`

The PAT needs `contents: write` on this repository and nothing else.

**This path is the known failure mode.** Between 2026-08-01 and 2026-08-04 no run fired
at all, which is consistent with the PAT being revoked during a token cleanup: the
external service kept POSTing, GitHub kept rejecting, and nothing in this repository
could tell. That is precisely why `schedule:` is now the primary trigger — an external
trigger that fails is silent, and a token that expires takes the forecast down without a
single red mark anywhere.

If you use this path, give the token an expiry you will actually notice, and check
cron-job.org's own failure log when runs go missing.

## When a run fails

The workflow notifies on failure rather than failing silently. To diagnose:

1. Actions → the failed run → the failing step.
2. **Logs expire.** They were already gone for the 2026-07-29 to 07-31 failures by the
   time anyone looked, which is why the cause of those runs was never established. If a
   failure matters, capture the log while it is still there.
3. A job that dies within a few seconds failed in setup or checkout, not in the
   pipeline. A job that dies later is the pipeline itself, and `pytest` locally will
   usually reproduce it.
