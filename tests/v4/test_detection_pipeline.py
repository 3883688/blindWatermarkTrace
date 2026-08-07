from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from trace_app.v4.deadlines import Deadline, DeadlineExceeded
from trace_app.v4.detection import (
    DetectionRequest,
    ResourceExhausted,
    ServiceUnavailable,
    V4DetectionService,
)
from trace_app.v4.domain import DetectionOutcome, OwnerScope
from trace_app.v4.keys import KeyRing
from trace_app.v4.models import ModelRegistryError
from watermark_v4.payload import AuthContext, CODEC_ID, encode_codeword
from watermark_v4.observation import CarrierEvidence, V4Observation


KEYS = KeyRing({"key": b"k" * 32}, "key")
SOURCE_HASH = b"s" * 32


def _record(trace: str, owner: int = 7):
    context = AuthContext(CODEC_ID, "key", owner, SOURCE_HASH, trace)
    return SimpleNamespace(
        id=uuid4(), source_group_id=UUID(int=1), owner_user_id=owner,
        trace_id=trace, codec=CODEC_ID, auth_tag=KEYS.sign(context), key_id="key",
        original_file_sha256=b"f" * 32, original_pixel_sha256=SOURCE_HASH,
    )


class Repo:
    def __init__(self, record=None, exact=()):
        self.record, self.exact, self.lookups = record, exact, 0

    def find_exact_file(self, scope, *, md5, sha256):
        return self.exact

    def find_record_by_auth_tag(self, scope, *, source_group_id, auth_tag):
        self.lookups += 1
        return self.record if self.record and self.record.source_group_id == source_group_id and self.record.auth_tag == auth_tag else None

    def find_records_for_group(self, scope, *, source_group_id):
        if self.record and self.record.source_group_id == source_group_id:
            return (self.record,)
        return ()


def _service(repo, events, *, groups=(UUID(int=1),), observe_tag=None):
    tag = observe_tag or (_record("TRACE").auth_tag)
    return V4DetectionService(
        repository=repo,
        key_ring=KEYS,
        decode_rgb=lambda content, deadline: events.append("decode") or object(),
        recall_groups=lambda scope, image, deadline: events.append("recall") or groups,
        confirm_group=lambda image, group, deadline: events.append(("geometry", group)) or object(),
        extract_observation=lambda image, confirmed, deadline: events.append(("observation", confirmed)) or encode_codeword(tag),
    )


def test_exact_hash_lookup_returns_before_decode() -> None:
    record, events = _record("EXACT"), []
    result = _service(Repo(exact=(record,)), events).detect(
        DetectionRequest(OwnerScope(7), b"image"), Deadline.after(10)
    )
    assert result.outcome is DetectionOutcome.SUCCESS and result.record is record
    assert events == []

    cross_owner = _record("CROSS", owner=8)
    denied_events = []
    denied = _service(Repo(exact=(cross_owner,)), denied_events).detect(
        DetectionRequest(OwnerScope(7), b"image"), Deadline.after(10)
    )
    assert denied.outcome is DetectionOutcome.NOT_FOUND


def test_group_pipeline_authenticates_only_after_geometry_and_hmac() -> None:
    record, events = _record("TRACE"), []
    repo = Repo(record)
    result = _service(repo, events, observe_tag=record.auth_tag).detect(
        DetectionRequest(OwnerScope(7), b"query"), Deadline.after(10)
    )
    assert result.outcome is DetectionOutcome.SUCCESS and result.record is record
    assert events[0:2] == ["decode", "recall"]
    assert events[2][0] == "geometry" and events[3][0] == "observation"
    assert repo.lookups == 1

    forged = _record("FORGED")
    forged.auth_tag = record.auth_tag
    miss = _service(Repo(forged), [], observe_tag=record.auth_tag).detect(
        DetectionRequest(OwnerScope(7), b"query"), Deadline.after(10)
    )
    assert miss.outcome is DetectionOutcome.NOT_FOUND


def test_group_pipeline_recovers_bit_sparse_damage_after_geometry() -> None:
    record = _record("CROPPED")
    damaged = bytearray(encode_codeword(record.auth_tag))
    for index in range(10):
        damaged[index] ^= 1 << (index % 2)
    evidence = CarrierEvidence(0, 4, 2, 0.8, 0.5)
    observation = V4Observation(
        bytes(damaged),
        (0.5,) * 16,
        (evidence, CarrierEvidence(1, 4, 2, 0.8, 0.5)),
        0.01,
    )
    service = V4DetectionService(
        repository=Repo(record),
        key_ring=KEYS,
        decode_rgb=lambda content, deadline: object(),
        recall_groups=lambda scope, image, deadline: (record.source_group_id,),
        confirm_group=lambda image, group, deadline: object(),
        extract_observation=lambda image, confirmed, deadline: observation,
    )

    result = service.detect(
        DetectionRequest(OwnerScope(7), b"cropped"), Deadline.after(10)
    )

    assert result.outcome is DetectionOutcome.SUCCESS
    assert result.record is record


def test_zero_and_multiple_authenticated_groups_do_not_expose_a_record() -> None:
    assert _service(Repo(None), []).detect(DetectionRequest(OwnerScope(7), b"q"), Deadline.after(10)).outcome is DetectionOutcome.NOT_FOUND
    record = _record("TRACE")
    result = _service(Repo(record), [], groups=(UUID(int=1), UUID(int=1)), observe_tag=record.auth_tag).detect(
        DetectionRequest(OwnerScope(7), b"q"), Deadline.after(10)
    )
    assert result.outcome is DetectionOutcome.SUCCESS

    other = _record("OTHER")
    other.source_group_id = UUID(int=2)
    records = {(record.source_group_id, record.auth_tag): record, (other.source_group_id, other.auth_tag): other}

    class MultiRepo(Repo):
        def find_record_by_auth_tag(self, scope, *, source_group_id, auth_tag):
            return records.get((source_group_id, auth_tag))

    ambiguous = V4DetectionService(
        repository=MultiRepo(), key_ring=KEYS,
        decode_rgb=lambda content, deadline: object(),
        recall_groups=lambda scope, image, deadline: (UUID(int=1), UUID(int=2)),
        confirm_group=lambda image, group, deadline: group,
        extract_observation=lambda image, group, deadline: encode_codeword(
            record.auth_tag if group == UUID(int=1) else other.auth_tag
        ),
    ).detect(DetectionRequest(OwnerScope(7), b"q"), Deadline.after(10))
    assert ambiguous.outcome is DetectionOutcome.AMBIGUOUS and ambiguous.record is None


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (DeadlineExceeded("test"), DetectionOutcome.TIMEOUT),
        (ResourceExhausted("full"), DetectionOutcome.RESOURCE_EXHAUSTED),
        (ServiceUnavailable("model"), DetectionOutcome.SERVICE_UNAVAILABLE),
        (ModelRegistryError("model"), DetectionOutcome.SERVICE_UNAVAILABLE),
    ],
)
def test_typed_failures_preserve_outcomes(error, outcome) -> None:
    service = V4DetectionService(
        repository=Repo(), key_ring=KEYS,
        decode_rgb=lambda content, deadline: (_ for _ in ()).throw(error),
        recall_groups=lambda *args: (), confirm_group=lambda *args: None,
        extract_observation=lambda *args: None,
    )
    assert service.detect(DetectionRequest(OwnerScope(7), b"q"), Deadline.after(10)).outcome is outcome
