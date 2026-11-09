import json

import pytest

from src.accuracy_reporting import generate_report
from src.v3_research import digest, save_json


def study(folder):
    choices = {"selections": {}, "primary_winners": {str(h): {"model": "sgd"} for h in (30, 60)},
               "primary_baselines": {str(h): {"model": "ridge"} for h in (30, 60)}}
    save_json(folder / "choices.json", choices)
    (folder / "choices.sha256").write_text(digest(folder / "choices.json"))
    save_json(folder / "final_completed.json", {"choices_sha256": digest(folder / "choices.json"),
              "candidates": 100, "event_proxies": 25})
    save_json(folder / "audit.json", {"status": "passed", "scope": "completed_final"})
    save_json(folder / "segments.json", {"final_sessions": 30})
    metrics = {}
    for h in (30, 60):
        for name, mae in (("selected_sgd", 2), ("baseline_ridge", 3), ("selected_gru", .1)):
            metrics[f"{name}_{h}_seed42"] = {"observed_mae": mae, "closed_mae": mae,
                "observed_sides": 180, "total_sides": 200, "rejected_sides": 5,
                "unknown_sides": 20, "observed_coverage": .9}
    save_json(folder / "final_metrics.json", metrics)
    save_json(folder / "primary_gates.json", {str(h): {"winner": "sgd", "strongest_default": "ridge",
        "relative_observed_mae_improvement": 1/3, "improvement_cents_95ci": [.5, 1.5],
        "passes_both": True, "observed_event_groups": 25} for h in (30, 60)})


def test_report_keeps_preselected_winner_even_when_another_final_model_scores_better(tmp_path):
    study(tmp_path)
    text = generate_report(tmp_path).read_text(encoding="utf-8")
    assert "| 30s | sgd | ridge | 2.0000 | 3.0000 |" in text
    assert "| 60s | sgd | ridge | 2.0000 | 3.0000 |" in text
    assert "selected_gru" in text  # Secondary result is retained, not selected post hoc.
    assert "not a lower bound on MAE" in text


def test_report_refuses_development_only_audit_and_changed_choices(tmp_path):
    study(tmp_path)
    save_json(tmp_path / "audit.json", {"status": "passed", "scope": "development"})
    with pytest.raises(ValueError, match="final-scope"):
        generate_report(tmp_path)
    save_json(tmp_path / "audit.json", {"status": "passed", "scope": "completed_final"})
    save_json(tmp_path / "choices.json", {"altered": True})
    with pytest.raises(ValueError, match="frozen choices"):
        generate_report(tmp_path)


def test_report_rejects_gate_changing_the_frozen_winner(tmp_path):
    study(tmp_path)
    gates = json.loads((tmp_path / "primary_gates.json").read_text())
    gates["30"]["winner"] = "gru"
    save_json(tmp_path / "primary_gates.json", gates)
    with pytest.raises(ValueError, match="frozen selection"):
        generate_report(tmp_path)
