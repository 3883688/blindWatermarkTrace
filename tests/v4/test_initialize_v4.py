from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.initialize_v4 import GuardedInitializer, validate_destructive_target
from trace_app.v4.startup import read_ready_marker


def test_destructive_targets_reject_roots_home_and_workspace(tmp_path: Path) -> None:
    for unsafe in (Path(tmp_path.anchor), Path.home(), tmp_path):
        with pytest.raises(ValueError):
            validate_destructive_target(unsafe, workspace=tmp_path)
    target = tmp_path / "uploads"
    target.mkdir()
    assert validate_destructive_target(target, workspace=tmp_path) == target.resolve()


def test_apply_requires_exact_confirmation_and_writes_marker_last(tmp_path: Path) -> None:
    events = []
    uploads = tmp_path / "workspace" / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "old.bin").write_bytes(b"old")
    backup = tmp_path / "backup"
    marker = tmp_path / "data" / "v4-ready.json"
    identity = ((1, "admin", "admin"), (7, "user", "operator"))
    workflow = GuardedInitializer(
        environment="test", database_name="trace_test", workspace=tmp_path / "workspace",
        upload_dir=uploads, backup_dir=backup, ready_marker=marker,
        dump_database=lambda path: path.write_bytes(b"database dump"),
        restore_database=lambda path: events.append("restore"),
        snapshot_identities=lambda: identity,
        clear_algorithm_data=lambda: events.append("clear"),
        create_schema=lambda: events.append("schema"),
        smoke_test=lambda: events.append("smoke"),
        rotate_key=lambda: "key-2026-07",
    )
    workflow.backup()
    with pytest.raises(ValueError):
        workflow.apply("yes")
    assert not marker.exists()

    workflow.apply("RESET-V4:test:trace_test")
    assert events == ["clear", "schema", "smoke"]
    assert read_ready_marker(marker) == {
        "schema_id": "v4", "model_id": "v4-models", "key_id": "key-2026-07"
    }
    assert "secret" not in marker.read_text(encoding="utf-8")


@pytest.mark.parametrize("failure", ["clear", "schema", "smoke"])
def test_failure_at_any_apply_phase_keeps_service_offline(tmp_path: Path, failure: str) -> None:
    uploads = tmp_path / "workspace" / "uploads"; uploads.mkdir(parents=True)
    marker = tmp_path / "ready.json"
    def phase(name):
        if name == failure: raise RuntimeError(name)
    workflow = GuardedInitializer(
        environment="test", database_name="db", workspace=tmp_path / "workspace",
        upload_dir=uploads, backup_dir=tmp_path / "backup", ready_marker=marker,
        dump_database=lambda path: path.write_bytes(b"dump"), restore_database=lambda path: None,
        snapshot_identities=lambda: ((1, "admin", "admin"),),
        clear_algorithm_data=lambda: phase("clear"), create_schema=lambda: phase("schema"),
        smoke_test=lambda: phase("smoke"), rotate_key=lambda: "key-id",
    )
    workflow.backup()
    with pytest.raises(RuntimeError):
        workflow.apply("RESET-V4:test:db")
    assert not marker.exists()
