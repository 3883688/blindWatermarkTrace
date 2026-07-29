from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from trace_app.imaging import io as image_io
from trace_app.imaging.io import _validate_public_http_url
from trace_app.v4.deadlines import Deadline, DeadlineExceeded
from trace_app.v4.uploads import UploadLimitExceeded, decode_image_unbounded, stream_upload


class ChunkedUpload:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.offset = 0
        self.requested_sizes: list[int] = []

    async def read(self, size: int) -> bytes:
        self.requested_sizes.append(size)
        chunk = self.content[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


def test_upload_is_streamed_in_bounded_chunks(tmp_path: Path) -> None:
    upload = ChunkedUpload(b"a" * (2 * 1024 * 1024 + 7))
    staged = asyncio.run(
        stream_upload(upload, temp_root=tmp_path, max_bytes=3 * 1024 * 1024)
    )
    try:
        assert staged.byte_size == len(upload.content)
        assert max(upload.requested_sizes) == 1024 * 1024
        assert staged.path.read_bytes() == upload.content
    finally:
        staged.cleanup()


def test_upload_limit_removes_partial_private_file(tmp_path: Path) -> None:
    upload = ChunkedUpload(b"x" * 32)
    with pytest.raises(UploadLimitExceeded):
        asyncio.run(stream_upload(upload, temp_root=tmp_path, max_bytes=16))
    assert list(tmp_path.rglob("*")) == []


def test_v4_worker_decode_has_no_fixed_pixel_ceiling(monkeypatch) -> None:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "black").save(buffer, format="PNG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    image = decode_image_unbounded(buffer.getvalue())
    assert image.size == (2, 2)


def test_public_url_validation_returns_only_pinnable_addresses() -> None:
    addresses = _validate_public_http_url(
        "https://images.example.test/a.png",
        resolve_host=lambda *_args, **_kwargs: [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ],
    )
    assert addresses == ("93.184.216.34",)


def test_default_remote_fetch_connects_to_the_validated_ip(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class Response:
        status = 200
        headers = {"content-type": "image/png"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit: int) -> bytes:
            return b"png"

    def pinned_open(url: str, connect_ip: str, *, timeout: float):
        calls.append((url, connect_ip))
        return Response()

    monkeypatch.setattr(image_io, "_open_pinned_url", pinned_open)
    assert image_io.fetch_remote_image_bytes(
        "https://images.example.test/a.png",
        resolve_host=lambda *_args, **_kwargs: [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ],
    )[0] == b"png"
    assert calls == [
        ("https://images.example.test/a.png", "93.184.216.34")
    ]


def test_pinned_connection_closes_when_request_fails(monkeypatch) -> None:
    closed: list[bool] = []

    class BrokenConnection:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def request(self, *_args, **_kwargs) -> None:
            raise OSError("connect failed")

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(image_io.http.client, "HTTPConnection", BrokenConnection)
    with pytest.raises(OSError):
        image_io._open_pinned_url(
            "http://images.example.test/a.png",
            "93.184.216.34",
            timeout=1,
        )
    assert closed == [True]


def test_remote_fetch_obeys_shared_absolute_deadline() -> None:
    now = [10.0]
    deadline = Deadline.after(2, clock=lambda: now[0])

    class SlowResponse:
        status = 200
        headers = {"content-type": "image/png"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit: int) -> bytes:
            now[0] += 3
            return b"slow bytes"

    timeouts: list[float] = []

    def open_url(_request, *, timeout: float):
        timeouts.append(timeout)
        return SlowResponse()

    with pytest.raises(DeadlineExceeded):
        image_io.fetch_remote_image_bytes(
            "https://images.example.test/a.png",
            deadline=deadline,
            resolve_host=lambda *_args, **_kwargs: [
                (2, 1, 6, "", ("93.184.216.34", 443))
            ],
            open_url=open_url,
        )
    assert timeouts == [2.0]


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "http://user@93.184.216.34/a", "http://127.0.0.1/a"],
)
def test_non_public_targets_are_rejected(url: str) -> None:
    with pytest.raises(Exception):
        _validate_public_http_url(
            url,
            resolve_host=lambda *_args, **_kwargs: [
                (2, 1, 6, "", ("127.0.0.1", 80))
            ],
        )
