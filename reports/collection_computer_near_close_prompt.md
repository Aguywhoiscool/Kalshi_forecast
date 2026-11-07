# Prompt for Codex on the collection computer

Continue the approved near-close collection plan in this machine's existing
Kalshi project. Research changes were prepared and tested on the other computer;
the live collector here has NOT been updated by that work.

First read `reports/near_close_collection_handoff.md` from the current origin.
Inspect the local Git diff, collector launch method, current process command
lines, activity flags and observer cache/cursor changes. Preserve this machine's
credentials, Syncthing settings, activity settings and unrelated local edits.
Do not blindly pull over local modifications or replace the entire collector.
Fetch/read the research commits and apply the smallest compatible patch.

Relevant source changes include `src/feature_sampler.py`,
`src/websocket_client.py`, and `src/market_ranker.py`. The ranker must accept the
new optional `min_minutes_to_close` argument; its default remains conservative.
Supporting tests are `tests/test_close_metadata.py` and the background-review
tests. The retrospective audit helper imported by the new test must also be
available. Do not import a test that contacts a live account during collection.

Required behavior:

1. NEW feature sessions include `scheduled_close_time`,
   `close_time_observed_at`, and `seconds_to_scheduled_close`. Populate them only
   after the existing background REST review supplies a schedule; leave earlier
   rows blank. No new network calls in the sampling loop and no rewriting data.
2. Launch with `COLLECTOR_MIN_CLOSE_MINUTES=10`. Pass the setting to initial and
   fallback REST discovery. Keep the existing ten-minute lifecycle cutoff and
   existing subscription rotation. Do not stitch sessions across reconnects.
3. Preserve both enabled activity flags and all cache/cursor changes. Note that
   observer candidate selection can remain more conservative than REST fallback;
   do not expand the observer's policy without reviewing its separate filters.
4. No trades, order submissions, reserved outcome evaluation or dataset/model
   overwrites. Do not start a second recorder.

The user approved proceeding with this plan. Before a deliberate restart, finish
the patch, offline checks and source backup, and confirm the exact collector
runtime/launcher process IDs and paths. Follow this machine's established restart
requirements. If a distinct approval remains required by local instructions,
present the concrete tested patch and restart command for that final approval;
do not ask the user to repeat authorization that already applies.

Verify at least ten minutes after rollout: fresh snapshots, completed status
reviews, growing raw/feature files, new CSV headers, populated schedule values,
observation times no later than sample timestamps, and roughly one-second
sampling without status-review stalls. Distinguish reconnect gaps by session.
Record evidence, exact paths and rollout time, current process IDs and rollback
instructions. Report explicitly whether rollout occurred. Leave recording running.

Send the user a concise handoff for the research computer. Code alone does not
prove that new metadata is being collected; include observed sample evidence.
