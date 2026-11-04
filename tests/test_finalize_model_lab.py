import json

import pytest

import src.finalize_model_lab as finalizer
from src.movement_labels import LABEL_POLICY


def test_finalizer_publishes_pointer_only_after_audit_passes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "artifacts/run"
    run.mkdir(parents=True)
    (run / "status.json").write_text('{"stage":"complete"}')

    def audit(benchmark, output):
        assert not (tmp_path / "artifacts/model_lab_latest.json").exists()
        output.mkdir()
        (output / "audit.json").write_text(json.dumps({"label_policy": LABEL_POLICY,
            "status": "Corrected-label benchmark audit passed; exploratory"}))

    monkeypatch.setattr(finalizer, "audit", audit)
    finalizer.finalize(run)
    pointer = json.loads((tmp_path / "artifacts/model_lab_latest.json").read_text())
    assert pointer["label_policy"] == LABEL_POLICY
    assert (run / "verified_complete.json").exists()


def test_finalizer_does_not_publish_failed_audit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "artifacts/run"
    run.mkdir(parents=True)
    (run / "status.json").write_text('{"stage":"complete"}')

    def audit(*args):
        raise ValueError("Source hash mismatch")

    monkeypatch.setattr(finalizer, "audit", audit)
    with pytest.raises(ValueError, match="hash mismatch"):
        finalizer.finalize(run)
    assert not (tmp_path / "artifacts/model_lab_latest.json").exists()
    assert (run / "verification_failed.json").exists()


def test_failed_training_prevents_audit_or_pointer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "artifacts/run"
    run.mkdir(parents=True)
    (run / "status.json").write_text('{"stage":"failed"}')
    with pytest.raises(RuntimeError, match="Research run failed"):
        finalizer.finalize(run)
    assert not (tmp_path / "artifacts/model_lab_latest.json").exists()
