"""Exact-first, source-group-first authenticated V4 detection."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence
from uuid import UUID

from reedsolo import RSCodec, ReedSolomonError

from trace_app.v4.deadlines import Deadline, DeadlineExceeded
from trace_app.v4.domain import DetectionOutcome, DetectionResult, OwnerScope
from trace_app.v4.features import FeatureEnvelopeError
from trace_app.v4.keys import KeyRing
from trace_app.v4.models import ModelRegistryError
from watermark_v4.payload import AuthContext, CODEC_ID, decode_candidate_codeword
from watermark_v4.observation import V4Observation


class ResourceExhausted(RuntimeError):
    pass


class ServiceUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DetectionRequest:
    scope: OwnerScope
    content: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("detection content must be non-empty bytes")


class DetectionRepository(Protocol):
    def find_exact_file(self, scope: OwnerScope, *, md5: bytes, sha256: bytes) -> Sequence[object]: ...
    def find_record_by_auth_tag(
        self, scope: OwnerScope, *, source_group_id: UUID, auth_tag: bytes
    ) -> object | None: ...
    def find_records_for_group(
        self, scope: OwnerScope, *, source_group_id: UUID
    ) -> Sequence[object]: ...


def decode_observed_auth_tag(codeword: bytes) -> bytes | None:
    if not isinstance(codeword, bytes) or len(codeword) != 16:
        return None
    try:
        decoded, _corrected, _errata = RSCodec(8, nsize=16).decode(codeword)
    except (ReedSolomonError, ValueError, TypeError, IndexError):
        return None
    tag = bytes(decoded)
    return tag if len(tag) == 8 else None


def _unique_records(records: Sequence[object]) -> tuple[object, ...]:
    unique: dict[object, object] = {}
    for record in records:
        unique[getattr(record, "id")] = record
    return tuple(unique.values())


class V4DetectionService:
    def __init__(
        self,
        *,
        repository: DetectionRepository,
        key_ring: KeyRing,
        decode_rgb: Callable[[bytes, Deadline], object],
        recall_groups: Callable[[OwnerScope, object, Deadline], Sequence[UUID]],
        confirm_group: Callable[[object, UUID, Deadline], object | None],
        extract_observation: Callable[[object, object, Deadline], bytes | None],
    ) -> None:
        self.repository = repository
        self.key_ring = key_ring
        self.decode_rgb = decode_rgb
        self.recall_groups = recall_groups
        self.confirm_group = confirm_group
        self.extract_observation = extract_observation

    def detect(self, request: DetectionRequest, deadline: Deadline) -> DetectionResult:
        try:
            digest_md5 = hashlib.md5(request.content).digest()
            digest_sha256 = hashlib.sha256(request.content).digest()
            exact = _unique_records(
                tuple(
                    record
                    for record in self.repository.find_exact_file(
                    request.scope, md5=digest_md5, sha256=digest_sha256
                    )
                    if self._record_in_scope(request.scope, record)
                )
            )
            if len(exact) == 1:
                return DetectionResult(DetectionOutcome.SUCCESS, exact[0])
            if len(exact) > 1:
                return DetectionResult(DetectionOutcome.AMBIGUOUS)

            deadline.check("detection_decode")
            image = self.decode_rgb(request.content, deadline)
            deadline.check("detection_recall")
            group_ids = tuple(self.recall_groups(request.scope, image, deadline))[:40]
            authenticated: list[object] = []
            for group_id in group_ids:
                deadline.check("detection_geometry")
                confirmed = self.confirm_group(image, group_id, deadline)
                if confirmed is None:
                    continue
                deadline.check("detection_observation")
                observed = self.extract_observation(image, confirmed, deadline)
                codeword = (
                    observed.observed_codeword
                    if isinstance(observed, V4Observation)
                    else observed
                )
                tag = None if codeword is None else decode_observed_auth_tag(codeword)
                if codeword is None:
                    continue
                if tag is not None:
                    record = self.repository.find_record_by_auth_tag(
                        request.scope, source_group_id=group_id, auth_tag=tag
                    )
                    if record is not None and self._verify_record(
                        request.scope, record, tag
                    ):
                        authenticated.append(record)
                        continue
                if not isinstance(observed, V4Observation):
                    continue
                for record in self.repository.find_records_for_group(
                    request.scope, source_group_id=group_id
                ):
                    candidate_tag = getattr(record, "auth_tag", None)
                    if not isinstance(candidate_tag, bytes):
                        continue
                    decoded = decode_candidate_codeword(
                        codeword,
                        candidate_tag,
                        observed.byte_confidences,
                    )
                    if decoded is not None and self._verify_record(
                        request.scope, record, candidate_tag
                    ):
                        authenticated.append(record)
            unique = _unique_records(authenticated)
            if len(unique) == 1:
                return DetectionResult(DetectionOutcome.SUCCESS, unique[0])
            if len(unique) > 1:
                return DetectionResult(DetectionOutcome.AMBIGUOUS)
            return DetectionResult(DetectionOutcome.NOT_FOUND)
        except DeadlineExceeded:
            return DetectionResult(DetectionOutcome.TIMEOUT)
        except ResourceExhausted:
            return DetectionResult(DetectionOutcome.RESOURCE_EXHAUSTED)
        except ServiceUnavailable:
            return DetectionResult(DetectionOutcome.SERVICE_UNAVAILABLE)
        except (ModelRegistryError, FeatureEnvelopeError):
            return DetectionResult(DetectionOutcome.SERVICE_UNAVAILABLE)

    @staticmethod
    def _record_in_scope(scope: OwnerScope, record: object) -> bool:
        owner_id = getattr(record, "owner_user_id", None)
        return scope.query_owner_id is None or owner_id == scope.query_owner_id

    def _verify_record(self, scope: OwnerScope, record: object, tag: bytes) -> bool:
        owner_id = getattr(record, "owner_user_id")
        if not self._record_in_scope(scope, record):
            return False
        record_tag = getattr(record, "auth_tag", None)
        if (
            getattr(record, "codec", None) != CODEC_ID
            or not isinstance(record_tag, bytes)
            or not hmac.compare_digest(record_tag, tag)
        ):
            return False
        try:
            context = AuthContext(
                CODEC_ID,
                getattr(record, "key_id"),
                owner_id,
                getattr(record, "original_pixel_sha256"),
                getattr(record, "trace_id"),
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return self.key_ring.verify(context, tag)


__all__ = (
    "DetectionRequest",
    "ResourceExhausted",
    "ServiceUnavailable",
    "V4DetectionService",
    "decode_observed_auth_tag",
)
