# Hourly research readiness audit installed

Task: `KalshiResearch-ReadinessHourly`, on the research computer only.
Action: project `.venv\Scripts\pythonw.exe -m src.readiness_audit`, with the
research project as working directory. Trigger interval: `PT1H` (hourly).
Verified Task Scheduler last result: **0** after the first scheduled execution.
Next run at verification: 2026-11-07 22:38 EST. The task remains registered.

The task runs as the logged-in user at limited privilege, hidden, with
start-when-available and no wake or power-policy changes. It does not run while
the computer is asleep or the user is logged out. A missed run can catch up when
available. Scheduler ignores overlapping instances; an OS file lock also guards
manual/task overlap. No Codex session needs to remain active.

## Audit and current observations

Reads stable, completed synced feature CSVs after the targeted rollout at
2026-11-07 04:01:14 UTC. Excludes newest/recent/changing files and reserved periods.
Checks quote structure, schedule knowledge times, countdown accuracy and full
continuous 60-observation histories. No sessions are stitched and signals are
spaced by 120 seconds within each market/session. No future return labels are
calculated. Unchanged files use an incremental local cache.

The initial manual report found 610,099 verified schedule rows, 173,518 near-close
rows and 1,416 usable near-close windows across 176 sessions and 146 event proxies
over two UTC dates. It found zero invalid metadata rows and zero file errors.
These are timestamped initial observations, not permanent current counts.
Read the latest local report for current results.

Outputs (local, not published in the exchange):
- `artifacts/readiness/LATEST.md`: human-readable screen and limitations.
- `artifacts/readiness/status.json`: full counters and reasons.
- `artifacts/readiness/last_run.json`: latest run success/failure and timestamp.
- `artifacts/readiness/cache.json`: incremental scan results.

Pilot-screen targets are 1,000 windows, 50 sessions, 20 event proxies and three
UTC dates, with no metadata/file errors and completed samples within an hour.
These are heuristic review criteria, not statistical power or profitability
requirements. At initial verification the report said collecting because only
two UTC dates were represented. A ready status never triggers automatic training.
Event counts use a ticker-prefix proxy and do not prove independent events.

## Validation and source handling

227 local tests passed, 1 integration test skipped. New tests cover schedule
availability, future knowledge, countdown errors, gaps, session boundaries,
reserved sessions, pilot gates and overlapping-run protection. Windows task
action/trigger and successful exit code were read back after registration.
No network calls, trades, training, collector changes, remote restarts or changes
to Syncthing configuration occur in this audit.

The exchange bootstrap changes from the previous task remain separate local
changes; this task does not commit them. Source/tests and this report are shared
as reviewed research context only. The collection computer should not install
this research scheduler or change its recorder because of this packet.

To run manually: `python -m src.readiness_audit` from the research workspace.
To register/repair locally: `scripts/register_readiness_audit.ps1` (reviewed local
PowerShell installer; not a packet auto-application instruction). To disable:
`Disable-ScheduledTask -TaskName 'KalshiResearch-ReadinessHourly'`.
