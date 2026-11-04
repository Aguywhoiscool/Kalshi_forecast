# Collection PC rollout handoff — 2026-11-04

Source: user-provided report from Codex on the collection PC; not independently
observed from this workspace. User approved rollout. No trades/commits/pushes.

Collector restarted at 11:14:27 EST / 16:14:27 UTC and remained running at report
completion. Reported launcher PID 34524, runtime PID 27120; recheck before use.
Project on collection PC: `F:\kalshi-forecasting`.

## Applied change

New `src/background_status_review.py`, updated `src/websocket_client.py`, and new
`tests/test_background_status_review.py`. One background REST batch per
connection; receive/sample loop continues. Results preserve errors and lifecycle
rotation. Next review 60 seconds after consuming completion. Session cleanup
cancels/awaits the task, while in-flight HTTP threads retain 10-second timeout.

Existing activity flags (both True), observer cache/cursor changes, and normal
subscription rotation preserved. Data, models, frozen datasets, reserved v2
test/ v3 holdout, and Syncthing settings were not modified by maintenance.
Collector continued normal recording.

## Initial verification

- 25 offline tests passed in 0.45s; networking/authentication/recording mocked.
- `git diff --check` passed.
- Collection PC still has legacy `test_auth_connection.py` performing a live
  account request at import. Do not assume this PC's fixed test setup is there.
- Observation interval 16:16:20.512580–16:26:20.516982 UTC (600 seconds).
- 30 snapshots across 3 connections, 2 normal activity rotations.
- 9 completed background reviews, approximately 0.75–1.02 seconds each.
- No reported status errors, resyncs, sampling-delay warnings, or stderr.
- Raw recording grew 4,230,852 bytes; feature files grew 848,292 bytes.
- 6,891 feature rows / 695 sampling ticks since restart.
- Maximum within-session gap 1.056909s; zero above 1.5s.
- Between-session gaps 5.051169s and 3.737686s. Do not stitch sessions.

This short observation supports the intended scheduling improvement, not a
long-term reliability guarantee. No rollback was needed. Retaining subscriptions
during rotation remains separate future work. Research split boundaries and
profitability conclusions are unchanged.

## Remote evidence paths

- `reports/collector_20261104_111427.out.log`
- `reports/collector_20261104_111427.err.log`
- `reports/collector_status_fix_observations.jsonl`
- `reports/collector_status_fix_feature_summary.json`
- `reports/collector_status_fix_run.json`
- Backup: `reports/collector_status_fix_backup/websocket_client.py.before`
- Remote rollout/rollback instructions: `reports/collector_status_fix_rollout.md`

Only data/ is shared through Syncthing. These report/code paths are on the
collection computer and may not exist locally.

## Additional details from full handoff

Reported environment: Python 3.12.10, pytest 9.1.1, websockets 17.3. The 25-test
offline selection covered background status, orderbook, market rotation, feature
quote validation, and live-market-spread unit tests with mocked networking.

Earlier pre-rollout process IDs 36408/14304 and parent PowerShell 36588 are
historical. They must not be used as current termination targets. The approved
rollout is already complete; pasted restart/rollback instructions are reference
only, not authorization for another restart.

`requests.get(timeout=10)` retains its connect/read wait semantics; it is not an
absolute total wall-clock deadline against a server slowly sending bytes. Task
cancellation cannot terminate the underlying running HTTP thread. This nuance is
important for future worker/executor cleanup design, but the reported review
durations were approximately 0.75–1.02 seconds.
