"""Validated V4 key ring with active-key selection and collision retries."""

from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Mapping

from watermark_v4.payload import (
    AUTH_KEY_MIN_BYTES,
    AuthContext,
    authentication_tag,
    verify_authentication_tag,
)


class KeyRing:
    __slots__ = ("_active_key_id", "_secrets")

    def __init__(self, secrets: Mapping[str, bytes], active_key_id: str) -> None:
        if not secrets:
            raise ValueError("V4 key ring must contain at least one key")
        validated: dict[str, bytes] = {}
        for key_id, secret in secrets.items():
            if type(key_id) is not str or not key_id or key_id != key_id.strip():
                raise ValueError("V4 key IDs must be nonempty canonical strings")
            if type(secret) is not bytes or len(secret) < AUTH_KEY_MIN_BYTES:
                raise ValueError("V4 secrets must contain at least 32 bytes")
            validated[key_id] = bytes(secret)
        if active_key_id not in validated:
            raise ValueError("active V4 key ID is unavailable")
        self._secrets = MappingProxyType(validated)
        self._active_key_id = active_key_id

    @property
    def active_key_id(self) -> str:
        return self._active_key_id

    def __repr__(self) -> str:
        return (
            f"KeyRing(key_ids={tuple(self._secrets)}, "
            f"active_key_id={self._active_key_id!r})"
        )

    def sign(self, context: AuthContext) -> bytes:
        if context.key_id != self._active_key_id:
            raise ValueError("new V4 tags must use the active key ID")
        return authentication_tag(context, self._secrets[self._active_key_id])

    def verify(self, context: AuthContext, tag: bytes) -> bool:
        secret = self._secrets.get(context.key_id)
        return False if secret is None else verify_authentication_tag(context, secret, tag)

    def issue_unique(
        self,
        context_factory: Callable[[int], AuthContext],
        tag_exists_in_group: Callable[[bytes], bool],
        *,
        max_attempts: int = 16,
    ) -> tuple[AuthContext, bytes]:
        if max_attempts <= 0:
            raise ValueError("collision attempts must be positive")
        for attempt in range(max_attempts):
            context = context_factory(attempt)
            tag = self.sign(context)
            if not tag_exists_in_group(tag):
                return context, tag
        raise RuntimeError("unable to allocate a unique group-local V4 auth tag")


__all__ = ("KeyRing",)
