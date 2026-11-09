# Direct and residual neural return study

This round compares GRU, a causal temporal convolutional network, and a feedforward network using the same frozen development observations. Each architecture predicts net YES/NO returns directly or predicts a correction to the constant-book forecast, `-spread_in_cents - 4`. Interval Huber losses with 1-cent and 5-cent thresholds and interval absolute error retain unknown outcomes as intervals.

The search contains 18 configurations per horizon and three chronological folds: 108 actual fits. Scaling and diagnostic quartiles use only each fold's fit observations. Within-training calibration chooses checkpoints and applies the same fixed, one-pass interval-projection correction. Each architecture's development-selected model contributes equally to a candidate ensemble; development chooses the primary model before fresh-final evaluation. Development selection creates optimism in reported development comparisons.

GRU layers and dropout come from the preceding study's development choices. TCN uses three residual causal convolution blocks with 32 channels, kernel size 3, dilations 1/2/4, dropout 0.1, and average pooling. MLP uses the existing 50 summaries of 60 observations, hidden widths 128/64, and dropout 0.1. All use Adam at 0.001, batches of 256, at most 30 epochs, and patience 5. Network outputs and interval losses use dollars; stored predictions and reports use cents.

The previous 6,054-observation development cohort is preserved. The preceding final period remains consumed and excluded from this round's development and final evaluation. A new source snapshot contains 2,866 files and 9,097,402 rows. Its declared potential final segment had 59 sessions and 124 metadata-only event proxies, beginning no earlier than 2026-11-09 13:28:12.778966 UTC, 70 seconds after the consumed final's end. At declaration, actual common-history and observed-outcome group requirements were unverified; completed counts are in the results report.

Choices and all 18 refit artifacts (three architectures, two horizons, three seeds) are hashed before any fresh final outcomes are aligned. Seed 42 is primary; seeds 7 and 101 measure robustness. The primary must improve observed-side MAE by at least 5% with a positive paired event-proxy bootstrap confidence interval against both the frozen previous GRU and the strongest development-selected default, including the no-change forecast. At least 20 observed event proxies are required. These are exploratory screens, not a statistical power guarantee; unknown outcomes remain unidentified. No model is automatically promoted to the live dashboard and trading remains disabled.

Historical launch command, run after offline tests and independent review passed. Reusing the completed output is deliberately refused:

```powershell
.\.venv\Scripts\python.exe -u -m src.residual_accuracy_experiment --output artifacts/residual_accuracy_20261109 --source artifacts/residual_accuracy_source_20261109
```

Progress is written to `artifacts/residual_accuracy_20261109/status.json` and `artifacts/agent_status/research.log`. Frozen protocol, choices, fold predictions, scalers, checkpoint histories, calibration, diagnostics and final gates remain under the versioned run directory. Initial implementation tests include genuine tiny fits and reloads for all three architectures, TensorFlow loss-gradient checks, TCN prefix causality, exact OOF identity checks, and provenance tamper rejection. The full offline suite passed 297 tests with one skip; 21 Node tests passed. Independent prelaunch review found no remaining blocker.

Actual training started at 2026-11-09 20:56:07 UTC with `gru_30_direct_huber1`, fold 1. Standard output and errors are in `artifacts/residual_accuracy_20261109.log` and `.err.log`. Launcher PID was 27396 at launch. Fresh-final outcomes remained unopened until choices and refits were frozen. Initial tests alone did not establish model quality or profitability. No collector process, live prediction pointer or trading setting was changed.

Update: the run completed at 22:37:32 UTC. The final acceptance audit passed. See [the completed results](residual_accuracy_results_20261109.md) for the fresh-final comparisons, uncertainty and failed release screens. The launch-time PID above is historical and must not be treated as a running process.
