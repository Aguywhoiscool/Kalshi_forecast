"""Summarize completed, audited accuracy studies without selecting models."""

import argparse
import json
from pathlib import Path

from src.v3_research import digest


def read(folder, name):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def number(value, scale=1):
    return "unavailable" if value is None else f"{value*scale:.4f}"


def generate_report(folder):
    """Refuse incomplete/unaudited final results; preserve all saved decisions."""
    audit = read(folder, "audit.json")
    if audit.get("status") != "passed" or audit.get("scope") != "completed_final":
        raise ValueError("A completed final-scope audit must pass before reporting")
    completed = read(folder, "final_completed.json")
    if completed["choices_sha256"] != digest(folder / "choices.json"):
        raise ValueError("Final completion does not match frozen choices")
    if (folder / "choices.sha256").read_text().strip() != digest(folder / "choices.json"):
        raise ValueError("Frozen choices changed")
    segments = read(folder, "segments.json")
    choices = read(folder, "choices.json")
    metrics = read(folder, "final_metrics.json")
    gates = read(folder, "primary_gates.json")
    lines = ["# Return-size accuracy experiment", "",
        "Primary task: forecast quote-based YES/NO scenario returns in cents. Results do not establish executable fills or profitability.",
        "", f"Final cohort: {completed['candidates']:,} candidates, {completed['event_proxies']:,} ticker-derived event proxies.",
        f"New final segment: {segments['final_sessions']} sessions, reserved by timestamps before outcome scoring.",
        "", "## Preselected primary comparisons", "",
        "Both winner and strongest default were selected using development folds only. Seed 42 is primary; seeds 7/101 are robustness checks.", "",
        "| Horizon | Selected candidate | Strongest default | Candidate observed MAE (cents) | Default observed MAE (cents) | Relative improvement | Paired 95% interval (cents improvement) | Review gate |",
        "|---|---|---|---:|---:|---:|---|---|"]
    coverage = []
    for horizon in (30, 60):
        row = gates[str(horizon)]
        if (row["winner"] != choices["primary_winners"][str(horizon)]["model"] or
                row["strongest_default"] != choices["primary_baselines"][str(horizon)]["model"]):
            raise ValueError("Reported gate disagrees with the frozen selection")
        candidate = metrics[f"selected_{row['winner']}_{horizon}_seed42"]
        baseline = metrics[f"baseline_{row['strongest_default']}_{horizon}_seed42"]
        interval = row.get("improvement_cents_95ci")
        ci = "unavailable" if interval is None else "["+", ".join(number(v) for v in interval)+"]"
        lines.append(f"| {horizon}s | {row['winner']} | {row['strongest_default']} | {number(candidate['observed_mae'])} | {number(baseline['observed_mae'])} | {number(row.get('relative_observed_mae_improvement'),100)}% | {ci} | {'passed' if row.get('passes_both') else 'not passed'} |")
        coverage.append(f"{horizon}s observed-side coverage: {candidate['observed_sides']:,}/{candidate['total_sides']:,}; rejected sides {candidate['rejected_sides']:,}; unknown sides {candidate['unknown_sides']:,}. Bootstrap observed event groups: {row.get('observed_event_groups', row.get('groups', 0))}.")
    for item in coverage:
        lines.extend(["", item])
    lines += ["", "## Primary-seed and frozen-reference results", "",
        "| Model artifact | Observed MAE | Closed-only MAE | Observed coverage | Attempts | Scenario return lower/upper (cents) |",
        "|---|---:|---:|---:|---:|---|"]
    for name, row in metrics.items():
        if "_seed42" not in name and not name.startswith("frozen_old_"):
            continue
        execution = row.get("execution")
        returns = "not recorded" if execution is None else f"{number(execution['net_lower_total_cents'])} / {number(execution['net_upper_total_cents'])}"
        attempts = "—" if execution is None else str(execution["attempts"])
        lines.append(f"| {name} | {number(row['observed_mae'])} | {number(row['closed_mae'])} | {number(row['observed_coverage'],100)}% | {attempts} | {returns} |")
    lines += ["", "## Interpretation and remaining limits", "",
        "- A passed screen means at least 5% relative observed-MAE improvement, a positive paired interval and at least 20 observed event-proxy groups. These are exploratory per-horizon gates, not a power calculation or familywise significance claim.",
        "- Observed-side MAE describes known quote scenarios, not unknown returns. Missing candidates remain in the full-population interval/economic diagnostics.",
        "- Interval-Huber identifies a robust location only partially under missing outcomes; it is not a missingness correction or generally an expected-profit estimator.",
        "- Interval calibration and objective form a pipeline contrast. Improvements cannot be attributed purely to the loss function.",
        "- `full_lower_bound_mae` is error against the pessimistic return endpoint, not a lower bound on MAE. `interval_distance_mae` measures distance to the feasible interval.",
        "- Seeds 7/101 are robustness observations, not independent events or a rule for choosing a lucky seed. See seed_spread.json.",
        "- Time/series subgroup results are descriptive. Related markets can remain dependent despite ticker-derived event grouping and bootstrap resampling.",
        "- No live pointer, collector settings, real orders or reserved v2/v3 outcomes were changed. New feature schemas are offline-only pending integration/explanation checks.",
        "- This final period is now consumed for model assessment. Later studies need a newer final boundary; do not reuse it as fresh evidence.",
        "", "## Reproduction artifacts", "",
        "protocol.json, segments.json, folds/, ablations.json, search_results.json, choices.json/.sha256, refits.json, refit_manifest.json, final_metrics.json, primary_gates.json, gates.json, seed_spread.json, subgroup_diagnostics.csv and audit.json.",
        "", f"Frozen selections: {len(choices.get('selections', {}))}; audit scope: {audit['scope']}."]
    temporary = folder / "REPORT.md.tmp"
    temporary.write_text("\n".join(lines)+"\n", encoding="utf-8")
    temporary.replace(folder / "REPORT.md")
    return folder / "REPORT.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = Path.cwd().resolve(), args.output.resolve()
    if root not in output.parents or root / "data" in output.parents:
        parser.error("Use a workspace artifact directory outside data/")
    print(generate_report(output))
