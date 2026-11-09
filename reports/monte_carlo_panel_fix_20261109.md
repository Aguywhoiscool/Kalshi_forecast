# Monte Carlo panel repair

The user reported flat paths or a blank panel. Live explanation requests returned
HTTP 200 with all required fields. A captured real 60-second payload contained
48 preview paths of 61 observations, 2,048 scenarios and eight nonzero historical
increments; its preview prices ranged from 0.61 to 0.995. The sampled highlighted
path happened to remain unchanged for its first six simulated steps. This did
not demonstrate a Monte Carlo backend calculation failure.

Two concrete frontend issues were addressed:

1. The opening animation showed only time zero, where all paths coincide, and
   the complete fan appeared farther below the formula. The primary panel now
   opens with the full horizon fan above the formula. Play restarts at zero and
   stops at the endpoint. Actual path selection changes the visible highlight
   and random-draw caption. Duplicate fan controls were removed; scenario stats
   and the complete download remain.
2. Switching tabs after an expired/failed request cleared the message and left
   null data in an empty panel. Loading/error state now persists across tabs;
   renderer/asset failures produce a visible fallback rather than silent blank.

Flat/coincident scenarios are explained without adding fabricated price noise.
Backend models, random draws, predictions and collector behavior were unchanged.

Root checks: 244 Python tests passed, one integration test skipped; 21 Node DOM
tests passed. The actual captured-payload test ran locally, not skipped. That
optional test is skipped on machines lacking the local uncommitted debug capture;
hand-computable regressions still cover default fan, replay, path selection,
flat inputs, loading, expiry and visible rendering errors. Independent reviewer
also passed the 21 Node tests and inspected the real payload and fix. Automated
browser screenshot/layout QA remains unavailable; refresh the user's browser to
load the updated static scripts.
