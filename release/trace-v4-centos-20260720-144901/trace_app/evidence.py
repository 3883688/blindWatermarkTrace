from __future__ import annotations

from typing import Any


def evidence_uuid_fields(evidence_uuid: str) -> dict[str, str]:
    normalized = evidence_uuid.replace("-", "").upper()
    return {
        "evidence_uuid": normalized,
        "evidence_uuid_head": normalized[:4],
        "evidence_uuid_tail": normalized[-4:],
    }


def with_evidence_fields(
    result: dict[str, Any], record: dict[str, Any] | None
) -> dict[str, Any]:
    if not record:
        return result
    for key in ("evidence_uuid", "evidence_uuid_head", "evidence_uuid_tail"):
        if record.get(key) and not result.get(key):
            result[key] = record[key]
    return result
