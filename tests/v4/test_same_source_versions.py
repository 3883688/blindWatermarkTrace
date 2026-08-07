from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from trace_app.v4.deadlines import Deadline
from trace_app.v4.detection import DetectionRequest, V4DetectionService
from trace_app.v4.domain import DetectionOutcome, OwnerScope
from trace_app.v4.keys import KeyRing
from watermark_v4.payload import AuthContext, CODEC_ID, encode_codeword


@pytest.mark.parametrize("index", [0, 3, 137, 499, 731, 999])
def test_one_thousand_same_source_versions_use_one_indexed_lookup(index: int) -> None:
    keys = KeyRing({"key": b"k" * 32}, "key")
    source_hash, group_id = b"s" * 32, UUID(int=5)
    records = {}
    for value in range(1000):
        trace = f"TRACE-{value:04d}"
        context = AuthContext(CODEC_ID, "key", 7, source_hash, trace)
        tag = keys.sign(context)
        records[(group_id, tag)] = SimpleNamespace(
            id=uuid4(), source_group_id=group_id, owner_user_id=7, trace_id=trace,
            codec=CODEC_ID, auth_tag=tag, key_id="key", original_file_sha256=b"f" * 32,
            original_pixel_sha256=source_hash,
        )
    target = records[(group_id, list(records.keys())[index][1])]
    calls = {"lookup": 0, "decode": 0, "geometry": 0, "observation": 0}

    class Repository:
        def find_exact_file(self, *args, **kwargs): return ()
        def find_record_by_auth_tag(self, scope, *, source_group_id, auth_tag):
            calls["lookup"] += 1
            return records.get((source_group_id, auth_tag))

    service = V4DetectionService(
        repository=Repository(), key_ring=keys,
        decode_rgb=lambda content, deadline: calls.__setitem__("decode", calls["decode"] + 1) or object(),
        recall_groups=lambda scope, image, deadline: (group_id,),
        confirm_group=lambda image, group, deadline: calls.__setitem__("geometry", calls["geometry"] + 1) or object(),
        extract_observation=lambda image, confirmed, deadline: calls.__setitem__("observation", calls["observation"] + 1) or encode_codeword(target.auth_tag),
    )
    result = service.detect(DetectionRequest(OwnerScope(7), b"query"), Deadline.after(10))
    assert result.outcome is DetectionOutcome.SUCCESS and result.record.trace_id == target.trace_id
    assert calls == {"lookup": 1, "decode": 1, "geometry": 1, "observation": 1}
