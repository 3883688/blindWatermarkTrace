import json
import re
from typing import Any, Callable

from PIL import Image

from trace_app.config import BLOCK_SIZE, BLOCK_STRIDE, MAGIC


class PayloadTooLargeError(ValueError):
    pass


class WatermarkNotFoundError(LookupError):
    pass


def bits_from_bytes(data: bytes) -> list[int]:
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def bytes_from_bits(bits: list[int]) -> bytes:
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i : i + 8]:
            byte = (byte << 1) | bit
        result.append(byte)
    return bytes(result)


def lsb_bits_from_pixels(pixels: Any):
    for pixel in pixels:
        yield pixel[0] & 1
        yield pixel[1] & 1
        yield pixel[2] & 1


def valid_watermark_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    required_strings = ("id", "trace_id", "user_id", "mode", "created_at")
    for key in required_strings:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            return False
    if not re.fullmatch(r"[0-9a-f]{32}", payload["id"]):
        return False
    if not re.fullmatch(r"[0-9A-F]{32}", str(payload.get("evidence_uuid", ""))):
        return False
    if payload.get("mode") not in {"lsb", "dct", "dwt", "fft", "hybrid"}:
        return False
    return True


def packet_from_payload(payload: dict[str, Any]) -> bytes:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return MAGIC + len(data).to_bytes(4, "big") + data


def write_packet_to_pixels(pixels: list[tuple[int, int, int]], packet: bytes) -> list[tuple[int, int, int]]:
    bits = bits_from_bytes(packet)
    capacity = len(pixels) * 3
    if len(bits) > capacity:
        raise PayloadTooLargeError("图片尺寸过小，无法嵌入水印信息")

    out = []
    idx = 0
    for pixel in pixels:
        channels = list(pixel)
        for channel in range(3):
            if idx < len(bits):
                channels[channel] = (channels[channel] & 0xFE) | bits[idx]
                idx += 1
        out.append(tuple(channels))
    return out


def embed_lsb(
    image: Image.Image,
    payload: dict[str, Any],
    *,
    embed_block_lsb_fn: Callable[[Image.Image, dict[str, Any]], Image.Image] | None = None,
    write_packet_to_pixels_fn: Callable[
        [list[tuple[int, int, int]], bytes], list[tuple[int, int, int]]
    ] | None = None,
) -> Image.Image:
    rgb = image.convert("RGB")
    packet = packet_from_payload(payload)
    pixels = list(rgb.getdata())
    write_fn = write_packet_to_pixels_fn or write_packet_to_pixels
    rgb.putdata(write_fn(pixels, packet))
    block_fn = embed_block_lsb_fn or embed_block_lsb
    return block_fn(rgb, payload)


def iter_block_origins(width: int, height: int):
    if width < BLOCK_SIZE or height < BLOCK_SIZE:
        return
    for y in range(0, height - BLOCK_SIZE + 1, BLOCK_STRIDE):
        for x in range(0, width - BLOCK_SIZE + 1, BLOCK_STRIDE):
            yield x, y


def embed_block_lsb(image: Image.Image, payload: dict[str, Any]) -> Image.Image:
    rgb = image.convert("RGB")
    packet = packet_from_payload(payload)
    bits = bits_from_bytes(packet)
    width, height = rgb.size
    pixels = rgb.load()
    required = len(bits)
    block_capacity = BLOCK_SIZE * BLOCK_SIZE * 3
    if block_capacity < required:
        return rgb

    for ox, oy in iter_block_origins(width, height) or []:
        idx = 0
        for y in range(oy, oy + BLOCK_SIZE):
            for x in range(ox, ox + BLOCK_SIZE):
                channels = list(pixels[x, y])
                for channel in range(3):
                    if idx < required:
                        channels[channel] = (channels[channel] & 0xFE) | bits[idx]
                        idx += 1
                pixels[x, y] = tuple(channels)
                if idx >= required:
                    break
            if idx >= required:
                break
    return rgb


def decode_bits_from_pixels(pixels: Any) -> dict[str, Any] | None:
    bit_iter = lsb_bits_from_pixels(pixels)
    header_bits = []
    for _ in range(64):
        try:
            header_bits.append(next(bit_iter))
        except StopIteration:
            return None
    if len(header_bits) < 64:
        return None
    header = bytes_from_bits(header_bits)
    if len(header) < 8 or header[:4] != MAGIC:
        return None

    size = int.from_bytes(header[4:8], "big")
    if size <= 0 or size > 8192:
        return None
    payload_bits = []
    for _ in range(size * 8):
        try:
            payload_bits.append(next(bit_iter))
        except StopIteration:
            return None
    payload = bytes_from_bits(payload_bits)
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not valid_watermark_payload(decoded):
        return None
    return decoded


def extract_lsb(
    image: Image.Image,
    *,
    extract_full_lsb_fn: Callable[[Image.Image], dict[str, Any] | None] | None = None,
    extract_block_lsb_fn: Callable[[Image.Image], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    full_fn = extract_full_lsb_fn or extract_full_lsb
    block_fn = extract_block_lsb_fn or extract_block_lsb
    payload = full_fn(image)
    if payload:
        return payload
    payload = block_fn(image)
    if payload:
        return payload
    raise WatermarkNotFoundError("未检测到可识别的隐式水印")


def extract_full_lsb(image: Image.Image) -> dict[str, Any] | None:
    rgb = image.convert("RGB")
    payload = decode_bits_from_pixels(list(rgb.getdata()))
    if payload:
        return payload
    return None


def extract_block_lsb(image: Image.Image) -> dict[str, Any] | None:
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < BLOCK_SIZE or height < BLOCK_SIZE:
        return None
    pixels = rgb.load()
    step = 1 if width * height <= 250_000 else 8
    y_origins = range(0, height - BLOCK_SIZE + 1, step)
    x_origins = range(0, width - BLOCK_SIZE + 1, step)
    if width * height > 1_000_000:
        y_origins = range(0, height - BLOCK_SIZE + 1, BLOCK_STRIDE)
        x_origins = range(0, width - BLOCK_SIZE + 1, BLOCK_STRIDE)
    for oy in y_origins:
        for ox in x_origins:
            payload = decode_bits_from_pixels(
                pixels[x, y]
                for y in range(oy, oy + BLOCK_SIZE)
                for x in range(ox, ox + BLOCK_SIZE)
            )
            if payload:
                return payload
    return None
