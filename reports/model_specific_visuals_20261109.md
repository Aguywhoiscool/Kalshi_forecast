# Model-specific visualization redesign

Replaced the universal gradient teaching tab with distinct mechanisms for all
eight model types. The calculation tab also opens the corresponding mechanism,
so the explanation is graphical by default. Existing forecast provenance,
calibration/clipping calculations, stale/expired behavior and training replay
remain intact. No trained model or backend algorithm changed.

- Monte Carlo: replay actual path steps/draws and inspect diverging futures.
- Decision tree: show visited branches, comparisons and reached leaf.
- Random forest: actual tree outputs converge into the ensemble mean; glyphs
  are schematic and the diagram identifies its displayed subset.
- Gradient boosting: add actual stage corrections to cumulative forecasts.
- Ridge: signed standardized-feature contributions form a waterfall.
- SGD: replay real recorded optimizer updates and coefficient projections.
- GRU: labeled 64/32-unit recurrent architecture and 60 actual input observations;
  hidden states/gates are not invented. Sensitivity is not causal importance.
- Mean baseline: schematic target aggregation using the actual training means,
  with no fabricated individual target values.

Root verification: 244 local Python tests passed, one integration test skipped;
15 Node DOM interaction tests passed, covering all eight mechanism routes plus
Monte Carlo random-step values, boosting correction arithmetic and fixed-axis
SGD projection. Independent review caught and corrected an invented zero start
for boosting: its first displayed stage now includes the learned baseline and
first tree without asserting the baseline was zero. The live
server returned HTTP 200 for the page and the new referenced mechanism script.
These are not automated browser screenshot/layout checks; that environment
limitation remains. Backend predictions and model artifacts were not modified.

The collector remains unchanged. Research-side collection advice remains to
continue recording if storage/bandwidth permit: later observations can provide
unseen evaluation data while the trained snapshot stays frozen. More data alone
does not guarantee better forecasts or profitability.
