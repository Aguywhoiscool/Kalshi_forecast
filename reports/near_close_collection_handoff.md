# Near-close collection preparation

Prepared locally on 2026-11-06. Not deployed to the collection computer. Do not
overwrite that machine's activity flags, observer caches, credentials or .env.

Existing order-book CSVs remain usable. A separate retrospective audit joins
market close metadata to development timestamps; it never rewrites original
features or evaluates reserved outcomes. Retrospective schedules are marked
`asof_verified=false` and are not approved as causal training inputs. A schedule
retrieved today may differ from the one known then, especially for early closes.

The bounded local audit found schedules for 40/40 most recently observed
development markets. Among their 132,118 matched recorded rows, 4,782 were
10-60 minutes before the retrospectively reported close and 1,733 were 0-10
minutes before it. No rows were after that reported close. This sample is not
an estimate of coverage across all 3,478 development markets. Evidence is in
`artifacts/near_close_backfill_20261106/summary.json` and `schedules.jsonl`.

Collector changes:

- `src/feature_sampler.py` adds `scheduled_close_time`,
  `close_time_observed_at`, and `seconds_to_scheduled_close` to NEW session CSVs.
  The existing seven model inputs are unchanged. Fields remain blank until a
  successful existing background status review supplies the close schedule.
  Schedule updates apply only to future rows. No per-sample network calls added.
- `src/websocket_client.py` passes `COLLECTOR_MIN_CLOSE_MINUTES` to initial and
  fallback REST discovery. Default remains 60; set to 10 for near-close coverage.
  Existing status checks still rotate markets with fewer than ten minutes left.
  Activity observer filters and machine settings are preserved; activity-based
  replacement candidates may remain more conservative than REST discovery.

On the collection computer, review and transfer only the intended source diff
and tests. A naive replacement of websocket_client.py can lose local flags or
observer changes. Back up that machine's source and inspect its current diff
before applying the patch. Follow its existing approved restart workflow; verify
the actual process IDs and recording paths before stopping anything.

For a NEW collector process, set this in the launching PowerShell session:

```powershell
$env:COLLECTOR_MIN_CLOSE_MINUTES = '10'
# Launch with the collection computer's normal logging/restart procedure.
# python -u -m src.websocket_client
```

Changing an environment variable does not reconfigure the already-running
collector. Do not start a second recorder alongside it. After a deliberate
restart, confirm new session headers, successful status reviews, populated close
metadata, growing files and continued one-second sampling. Confirm no sessions
are stitched across reconnects. Existing datasets and models stay frozen.

Local offline verification:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

The full current suite gates live account integration behind an explicit opt-in;
inspect the other computer's tests before using full-suite collection there.

Bounded retrospective audit (public metadata GETs; no orders):

```powershell
python -u -m src.backfill_close_metadata --source artifacts/swing_source_all_20261106 --output artifacts/near_close_backfill_NEW --max-markets 40
```

This inventory excludes reserved sessions and timestamps, queries live market
metadata with historical fallback, discards result/settlement/price fields, and
reports descriptive coverage only. Trades/candles cannot be substituted for
missing one-second depth and imbalance history. If the collector never watched
a near-close interval, close metadata cannot reconstruct its order book.
