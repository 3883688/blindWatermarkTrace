from pathlib import Path
import os
import shutil
import subprocess


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "create_source_backup.ps1"


def _script_text() -> str:
    assert SCRIPT_PATH.is_file(), f"backup script is missing: {SCRIPT_PATH}"
    return SCRIPT_PATH.read_text(encoding="utf-8")


def test_backup_script_declares_fixed_artifact_names():
    script = _script_text()

    assert "trace-v3-source-20260713.zip" in script
    assert "trace-v3-source-20260713.manifest.txt" in script
    assert "SHA256SUMS" in script


def test_backup_script_declares_required_exclusions_and_archive_verification():
    script = _script_text()

    for excluded_name in (
        ".env",
        "data",
        "uploads",
        "output",
        "test_output",
        "__pycache__",
    ):
        assert excluded_name in script
    assert "Expand-Archive" in script


def test_backup_script_excludes_sensitive_and_runtime_paths_from_fixture(tmp_path):
    included_paths = (
        "watermark_auth.py",
        "src/keyframe.py",
        "docs/secret-handling.md",
        "frontend/assets/app.js",
    )
    excluded_paths = (
        ".env",
        ".env.example",
        "nested/.env.production",
        ".git/config",
        ".venv/pyvenv.cfg",
        "nested/venv/pyvenv.cfg",
        "nested/__pycache__/module.pyc",
        "nested/.pytest_cache/state",
        "nested/.mypy_cache/state",
        "nested/.ruff_cache/state",
        "nested/.cache/state",
        "nested/.playwright-cli/state",
        "nested/playwright/state.json",
        "nested/browser-profile/Preferences",
        "nested/browser_profiles/Default/Preferences",
        "nested/backups/old.txt",
        "nested/data/database.db",
        "nested/uploads/image.png",
        "nested/output/result.png",
        "nested/test_output/result.png",
        "nested/logs/server.txt",
        "nested/server.log",
        "config/credentials.json",
        "config/secrets.json",
        "config/service-account.json",
        "config/service_account.json",
        "certs/server.pem",
        "certs/server.key",
        "certs/server.p12",
        "certs/server.pfx",
        "nested/archive.zip",
    )
    for relative_path in included_paths + excluded_paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")

    completed = subprocess.run(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT_PATH),
            "-RootPath",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

    manifest_path = tmp_path / "backups" / "trace-v3-source-20260713.manifest.txt"
    manifest = set(manifest_path.read_text(encoding="utf-8").splitlines())
    assert set(included_paths) <= manifest
    assert not (set(excluded_paths) & manifest)


def _copy_script_into_repo(repo_path: Path) -> Path:
    copied_script = repo_path / "tools" / SCRIPT_PATH.name
    copied_script.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT_PATH, copied_script)
    return copied_script


def _assert_restricted_windows_acl(path: Path) -> None:
    acl_check = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            """
$acl = [System.IO.File]::GetAccessControl($env:ACL_TEST_PATH)
$expected = @(
    [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value,
    'S-1-5-18',
    'S-1-5-32-544'
) | Sort-Object
$rules = @($acl.GetAccessRules(
    $true,
    $true,
    [System.Security.Principal.SecurityIdentifier]
))
$actual = @($rules | ForEach-Object { $_.IdentityReference.Value } | Sort-Object)
if (-not $acl.AreAccessRulesProtected -or $rules.Count -ne 3 -or
    (Compare-Object $expected $actual) -or
    @($rules | Where-Object {
        $_.AccessControlType -ne 'Allow' -or $_.FileSystemRights -ne 'FullControl' -or $_.IsInherited
    }).Count -ne 0) {
    throw 'Restricted artifact ACL contract failed.'
}
""",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "ACL_TEST_PATH": str(path)},
    )
    assert acl_check.returncode == 0, acl_check.stderr or acl_check.stdout


def _apply_restricted_windows_acl(path: Path) -> None:
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            """
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
& icacls $env:ACL_TEST_PATH /inheritance:r /grant:r "*$sid`:(F)" '*S-1-5-18:(F)' '*S-1-5-32-544:(F)' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Test ACL setup failed.' }
""",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "ACL_TEST_PATH": str(path)},
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_backup_script_resolves_default_root_from_script_location(tmp_path):
    repo_path = tmp_path / "isolated-repo"
    copied_script = _copy_script_into_repo(repo_path)
    (repo_path / "normal_source.py").write_text("source", encoding="utf-8")

    completed = subprocess.run(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(copied_script),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    manifest_path = repo_path / "backups" / "trace-v3-source-20260713.manifest.txt"
    assert manifest_path.is_file()
    assert "normal_source.py" in manifest_path.read_text(encoding="utf-8").splitlines()
    for artifact_path in (repo_path / "backups").iterdir():
        _assert_restricted_windows_acl(artifact_path)


def test_failure_before_publish_preserves_existing_artifact_set(tmp_path):
    repo_path = tmp_path / "isolated-repo"
    copied_script = _copy_script_into_repo(repo_path)
    (repo_path / "normal_source.py").write_text("source", encoding="utf-8")
    backups_path = repo_path / "backups"
    backups_path.mkdir()
    sentinel_artifacts = {
        "trace-v3-source-20260713.zip": b"sentinel zip",
        "trace-v3-source-20260713.manifest.txt": b"sentinel manifest",
        "SHA256SUMS": b"sentinel checksum",
    }
    for name, content in sentinel_artifacts.items():
        artifact_path = backups_path / name
        artifact_path.write_bytes(content)
        _apply_restricted_windows_acl(artifact_path)

    completed = subprocess.run(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(copied_script),
            "-Force",
            "-FailBeforePublish",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Intentional pre-publication failure" in (completed.stderr + completed.stdout)
    for name, expected_content in sentinel_artifacts.items():
        assert (backups_path / name).read_bytes() == expected_content


def test_existing_artifacts_are_immutable_without_force(tmp_path):
    repo_path = tmp_path / "isolated-repo"
    copied_script = _copy_script_into_repo(repo_path)
    (repo_path / "normal_source.py").write_text("source", encoding="utf-8")
    backups_path = repo_path / "backups"
    backups_path.mkdir()
    sentinel_artifacts = {
        "trace-v3-source-20260713.zip": b"sentinel zip",
        "trace-v3-source-20260713.manifest.txt": b"sentinel manifest",
        "SHA256SUMS": b"sentinel checksum",
    }
    for name, content in sentinel_artifacts.items():
        (backups_path / name).write_bytes(content)

    completed = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(copied_script)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "use explicit -Force" in (completed.stderr + completed.stdout)
    for name, expected_content in sentinel_artifacts.items():
        assert (backups_path / name).read_bytes() == expected_content


def test_force_replaces_existing_artifact_set(tmp_path):
    repo_path = tmp_path / "isolated-repo"
    copied_script = _copy_script_into_repo(repo_path)
    (repo_path / "normal_source.py").write_text("source", encoding="utf-8")
    backups_path = repo_path / "backups"
    backups_path.mkdir()
    sentinel_artifacts = {
        "trace-v3-source-20260713.zip": b"sentinel zip",
        "trace-v3-source-20260713.manifest.txt": b"sentinel manifest",
        "SHA256SUMS": b"sentinel checksum",
    }
    for name, content in sentinel_artifacts.items():
        (backups_path / name).write_bytes(content)

    completed = subprocess.run(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(copied_script),
            "-Force",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    for name, sentinel_content in sentinel_artifacts.items():
        assert (backups_path / name).read_bytes() != sentinel_content
        _assert_restricted_windows_acl(backups_path / name)
    assert not list(backups_path.glob(".trace-v3-*-*"))


def test_acl_publication_failure_restores_existing_artifact_set(tmp_path):
    repo_path = tmp_path / "isolated-repo"
    copied_script = _copy_script_into_repo(repo_path)
    (repo_path / "normal_source.py").write_text("source", encoding="utf-8")
    backups_path = repo_path / "backups"
    backups_path.mkdir()
    sentinel_artifacts = {
        "trace-v3-source-20260713.zip": b"sentinel zip",
        "trace-v3-source-20260713.manifest.txt": b"sentinel manifest",
        "SHA256SUMS": b"sentinel checksum",
    }
    for name, content in sentinel_artifacts.items():
        artifact_path = backups_path / name
        artifact_path.write_bytes(content)
        _apply_restricted_windows_acl(artifact_path)

    completed = subprocess.run(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(copied_script),
            "-Force",
            "-FailAclPublication",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Intentional ACL publication failure" in (completed.stderr + completed.stdout)
    for name, expected_content in sentinel_artifacts.items():
        assert (backups_path / name).read_bytes() == expected_content
        _assert_restricted_windows_acl(backups_path / name)
    assert not list(backups_path.glob(".trace-v3-*-*"))


def test_publication_catch_restricts_restored_artifacts_before_cleanup():
    restoration_block = _script_text().split("$publishError = $_", maxsplit=1)[1]

    evidence_copy_index = restoration_block.index(
        "Copy-Item -LiteralPath $rollbackPath -Destination $restorePath"
    )
    staged_acl_index = restoration_block.index(
        "Apply-RestrictedArtifactAcl -Path $restorePath"
    )
    restore_move_index = restoration_block.index(
        "Move-Item -LiteralPath $restorePath"
    )
    final_acl_index = restoration_block.index(
        "Apply-RestrictedArtifactAcl -Path (Join-Path $backupDir $artifactName)"
    )
    assert evidence_copy_index < staged_acl_index < restore_move_index < final_acl_index
    assert "CRITICAL:" in restoration_block


def test_publication_uses_protected_same_volume_local_directories():
    script = _script_text()

    publish_path_index = script.index("$localPublishDir = Join-Path $backupDir")
    protect_publish_index = script.index(
        "Apply-RestrictedDirectoryAcl -Path $localPublishDir"
    )
    move_into_publish_index = script.index(
        "Move-Item -LiteralPath (Join-Path $tempArtifactsDir $artifactName)"
    )
    rollback_path_index = script.index("$localRollbackDir = Join-Path $backupDir")
    protect_rollback_index = script.index(
        "Apply-RestrictedDirectoryAcl -Path $localRollbackDir"
    )
    move_into_rollback_index = script.index(
        "Move-Item -LiteralPath $finalPath -Destination (Join-Path $localRollbackDir"
    )

    assert publish_path_index < protect_publish_index < move_into_publish_index
    assert rollback_path_index < protect_rollback_index < move_into_rollback_index
