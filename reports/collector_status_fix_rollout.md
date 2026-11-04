# Collector status-review fix

Prepared and tested locally. The user subsequently approved deployment on the
collection computer on 2026-11-04; see `collection_pc_rollout_20261104.md` for
the reported 10-minute observations. The steps below remain reference rollout
instructions, not a request to restart the already-updated collector.

## Change

`src/websocket_client.py` uses the new `src/background_status_review.py` helper.
The receive/sample loop schedules one public REST review batch, continues
streaming, and handles completed results on subsequent iterations. The next
review is due 60 seconds after handling completion. Failed requests remain
visible and closed markets still trigger lifecycle rotation. The session cancels
and awaits the background task on exit. Existing HTTP connect/read timeouts
remain in effect; they are not absolute total wall-clock deadlines. Python cannot
forcibly cancel an already-running HTTP thread.

No subscription-retention or dataset/schema changes are included.

## Verification

Five offline tests cover blocked requests while actual stream sampling continues,
single in-flight batch, visible failures, closed-market rotation, and cleanup.
Local suite: 130 passed, 1 skipped (live authentication integration test).

```powershell
python -m pytest tests/test_background_status_review.py -q -p no:cacheprovider
```

## Transfer and activate on collection computer

Syncthing currently shares only data/, so these source changes require a separate
manual transfer. Transfer outside the shared data/ directory. Compare the remote
collector with this version first: do not overwrite remote-only improvements.

1. Transfer `src/background_status_review.py`, the updated
   `src/websocket_client.py`, and `tests/test_background_status_review.py` to a
   staging directory on the collection computer.
2. At a planned restart, stop its collector with Ctrl+C and back up its existing
   `src/websocket_client.py` outside data/. Copy the staged files into the matching
   project paths, preserving any remote-specific changes after review.
3. Activate that computer's existing virtual environment. Run the targeted tests
   above. They require pytest-asyncio; if missing, use its normal reviewed
   dependency installation workflow rather than reinstalling all dependencies.
4. Restart the existing read-only collector from the project root:

```powershell
python -m src.websocket_client
```

5. Check fresh snapshots and feature-file growth. After the first minute, look for
   `LIVE MARKET STATUS REVIEW (background)` and a completion-duration message.
   Observe sampling delays for at least 10 minutes. This fix removes the explicit
   REST wait in the streaming coroutine; it does not guarantee all timing gaps
   disappear. Keep collecting enough before/after data for a comparable audit.

If a regression occurs, stop the collector, restore the backed-up collector file,
and restart. The unused new helper may remain on disk. Do not run two collectors
simultaneously as part of this rollout.
