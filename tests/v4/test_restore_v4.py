from pathlib import Path

import pytest

from tools.initialize_v4 import GuardedInitializer, create_upload_backup


def test_restore_recovers_database_and_upload_fixtures(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"; uploads = workspace / "uploads"; uploads.mkdir(parents=True)
    (uploads / "fixture.bin").write_bytes(b"before")
    restored = []
    workflow = GuardedInitializer(
        environment="test", database_name="db", workspace=workspace, upload_dir=uploads,
        backup_dir=tmp_path / "backup", ready_marker=tmp_path / "ready.json",
        dump_database=lambda path: path.write_bytes(b"dump"),
        restore_database=lambda path: restored.append(path.read_bytes()),
        snapshot_identities=lambda: ((1, "admin", "admin"),), clear_algorithm_data=lambda: None,
        create_schema=lambda: None, smoke_test=lambda: None, rotate_key=lambda: "key-id",
    )
    workflow.backup()
    (uploads / "fixture.bin").write_bytes(b"after")
    workflow.restore()
    assert restored == [b"dump"]
    assert (uploads / "fixture.bin").read_bytes() == b"before"
    assert not workflow.ready_marker.exists()


def test_upload_backup_rejects_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "uploads"; source.mkdir()
    outside = tmp_path / "outside.bin"; outside.write_bytes(b"secret")
    try:
        (source / "link.bin").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError):
        create_upload_backup(source, tmp_path / "uploads.tar")
