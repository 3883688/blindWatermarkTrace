import pytest

from tools.migrate_mysql_to_postgresql import (
    build_role_menu_rows,
    normalize_roles,
    validate_source_snapshot,
)


def test_normalize_roles_preserves_system_and_custom_roles_and_menu_labels() -> None:
    roles = {
        "admin": {"label": "管理员", "menus": ["role", "trace"]},
        "custom": {"label": "审计员", "menus": ["audit"]},
    }

    normalized = normalize_roles(roles)

    assert {
        key: {"label": value["label"], "is_system": value["is_system"]}
        for key, value in normalized.roles.items()
    } == {
        "admin": {"label": "管理员", "is_system": True},
        "custom": {"label": "审计员", "is_system": False},
    }
    assert normalized.menus["role"]["label"] == "角色管理"
    assert normalized.menus["role"]["sort_order"] == 3
    assert normalized.menus["audit"]["label"] == "audit"
    assert normalized.menus["audit"]["sort_order"] == 4
    assert build_role_menu_rows(normalized) == (
        ("admin", "role"),
        ("admin", "trace"),
        ("custom", "audit"),
    )


def test_validate_source_snapshot_rejects_unknown_user_role() -> None:
    with pytest.raises(ValueError, match="role"):
        validate_source_snapshot(
            images=[],
            roles={"admin": {"label": "管理员", "menus": []}},
            users={"alice": {"role": "missing", "password_hash": "hash"}},
            stats={},
        )


def test_validate_source_snapshot_accepts_custom_roles_and_json_stats() -> None:
    validate_source_snapshot(
        images=[{"id": "image-1", "position_index": 0, "data": {"id": "image-1"}}],
        roles={"audit": {"label": "审计员", "menus": ["trace"]}},
        users={"alice": {"role": "audit", "password_hash": "hash"}},
        stats={"detection_stats": {"attempts": 1, "successes": 1}},
    )
