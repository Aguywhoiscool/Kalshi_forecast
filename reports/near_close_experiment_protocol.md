# Controlled near-close experiment

Prepared before inspecting the full subgroup results. This is a future research
protocol, not a claim that schedules are verified or that a strategy is profitable.

## Inputs and eligibility

Preserve the existing seven book features and frozen models. A new time feature
is allowed only when each observation records a schedule and the time at which
it became known, with `close_time_observed_at <= timestamp`. Do not infer the
schedule known then from a later market response, ticker text or settlement.
Retrospective timing remains separately labeled descriptive metadata.

Keep sessions separate, require a valid continuous 60-observation history, and
space evaluated signals by 120 seconds per market/session. Exclude all existing
reserved v2/v3 sessions and periods. Missing future quotes stay in the cohort
with explicit return bounds; coverage never determines entry eligibility.

The first prospective target regime is 10-60 minutes before scheduled close,
consistent with the prepared collector cutoff. Existing under-ten-minute data
can be described separately, but the current collector rollout does not promise
new coverage in that band.

## Comparison

Use identical chronological fit/calibration/validation identities and fixed
70-second embargoes, with disjoint sessions and later validation. Do not reuse
the old, repeatedly inspected validation period as a final claim of improvement.
Keep any new final holdout untouched until the protocol and implementations are
frozen. Compare:

1. Seven features, trained on all eligible training observations.
2. Seven features, trained excluding near-close training observations.
3. Same eligible observations as (1), adding verified seconds-to-close.

Keep model settings, seeds, horizons and evaluation populations fixed across
variants. Fit every scaler and transform on training only. Use calibration only
for the predefined checkpoint/bias-correction rules. Report differing training
counts; (1) versus (2) is not automatically a pure feature-effect comparison.
The existing Monte Carlo method can remain a seven-feature baseline; it does not
become a trained time-aware model merely because a timing field exists.

## Outcomes

Report 30/60-second return forecasts and execution scenarios on both the complete
validation population and predefined 10-30, 30-60 and over-60-minute groups.
Retain the fixed one-second entry delay, one-contract depth checks, 1-cent entry
margin, and primary 4-cent cost buffer. The buffer is not actual historical fees.
Do not retune thresholds to make a subgroup positive.

Report prediction error, trade counts, known/missing execution outcomes, return
bounds, and coverage by session/event. Preserve global position blocking across
time groups. Quote-based outcomes are hypothetical fills. Treat related markets
as correlated; raw rows and overlapping windows do not constitute independent
evidence. A few positive trades cannot establish an edge.

## Current gate

The existing frozen snapshot predates the observed-schedule columns. The full
retrospective audit alone cannot approve variant (3). First verify a collection
computer rollout and future rows with trustworthy knowledge timestamps. Then
freeze the prospective dataset and assess whether there are enough separate
sessions/events to make the planned comparisons meaningful. Until then, subgroup
results are descriptive and live dashboard forecasts remain research-only.
