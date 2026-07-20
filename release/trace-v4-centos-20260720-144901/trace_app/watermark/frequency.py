import hashlib
from typing import Callable

import cv2
import numpy as np
import pywt
from PIL import Image

from trace_app.config import DCT_BLOCK, DCT_DELTA, DWT_DELTA, FFT_DELTA, ROBUST_MAGIC


def robust_pattern(bit_index: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(ROBUST_MAGIC + bit_index * 7919)
    coarse = rng.choice(np.array([-1, 1], dtype=np.int16), size=(4, 4))
    repeat = max(1, int(np.ceil(size / 4)))
    pattern = np.kron(coarse, np.ones((repeat, repeat), dtype=np.int16))
    return pattern[:size, :size]


def layer_seed(trace_id: str, layer: str) -> int:
    digest = hashlib.blake2b(f"{trace_id}:{layer}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def pseudo_random_signs(trace_id: str, layer: str, count: int) -> np.ndarray:
    rng = np.random.default_rng(layer_seed(trace_id, layer))
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=count)


def apply_dct_layer(image: Image.Image, trace_id: str) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    channel = arr[:, :, 1]
    height, width = channel.shape
    blocks_y = height // DCT_BLOCK
    blocks_x = width // DCT_BLOCK
    signs = pseudo_random_signs(trace_id, "dct", blocks_y * blocks_x)
    idx = 0
    for by in range(blocks_y):
        for bx in range(blocks_x):
            y = by * DCT_BLOCK
            x = bx * DCT_BLOCK
            block = channel[y : y + DCT_BLOCK, x : x + DCT_BLOCK]
            coeff = cv2.dct(block)
            coeff[3, 4] += signs[idx] * DCT_DELTA
            coeff[4, 3] += signs[idx] * DCT_DELTA
            channel[y : y + DCT_BLOCK, x : x + DCT_BLOCK] = cv2.idct(coeff)
            idx += 1
    arr[:, :, 1] = channel
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def dct_layer_score(image: Image.Image, trace_id: str) -> float:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    channel = arr[:, :, 1]
    height, width = channel.shape
    blocks_y = height // DCT_BLOCK
    blocks_x = width // DCT_BLOCK
    count = blocks_y * blocks_x
    if count < 16:
        return 0.0
    signs = pseudo_random_signs(trace_id, "dct", count)
    values = []
    idx = 0
    for by in range(blocks_y):
        for bx in range(blocks_x):
            y = by * DCT_BLOCK
            x = bx * DCT_BLOCK
            coeff = cv2.dct(channel[y : y + DCT_BLOCK, x : x + DCT_BLOCK])
            values.append((coeff[3, 4] + coeff[4, 3]) * signs[idx])
            idx += 1
    values = np.array(values, dtype=np.float32)
    return float(max(0.0, values.mean() / (values.std() + 1e-6)))


def apply_dwt_layer(image: Image.Image, trace_id: str) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    channel = arr[:, :, 0]
    coeffs = pywt.dwt2(channel, "haar")
    ll, (lh, hl, hh) = coeffs
    signs = pseudo_random_signs(trace_id, "dwt", lh.size).reshape(lh.shape)
    lh = lh + signs * DWT_DELTA
    rebuilt = pywt.idwt2((ll, (lh, hl, hh)), "haar")
    arr[: rebuilt.shape[0], : rebuilt.shape[1], 0] = rebuilt[: arr.shape[0], : arr.shape[1]]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def dwt_layer_score(image: Image.Image, trace_id: str) -> float:
    channel = np.array(image.convert("RGB"), dtype=np.float32)[:, :, 0]
    _, (lh, _, _) = pywt.dwt2(channel, "haar")
    signs = pseudo_random_signs(trace_id, "dwt", lh.size).reshape(lh.shape)
    values = (lh * signs).ravel()
    return float(max(0.0, values.mean() / (values.std() + 1e-6)))


def fft_pattern(shape: tuple[int, int], trace_id: str) -> np.ndarray:
    height, width = shape
    rng = np.random.default_rng(layer_seed(trace_id, "fft"))
    pattern = np.zeros((height, width), dtype=np.float32)
    center_y, center_x = height // 2, width // 2
    radius_min = max(12, min(height, width) // 10)
    radius_max = max(radius_min + 4, min(height, width) // 4)
    for _ in range(96):
        angle = rng.uniform(0, np.pi)
        radius = rng.integers(radius_min, radius_max)
        y = int(round(center_y + np.sin(angle) * radius))
        x = int(round(center_x + np.cos(angle) * radius))
        y2 = int(round(center_y - np.sin(angle) * radius))
        x2 = int(round(center_x - np.cos(angle) * radius))
        if 0 <= y < height and 0 <= x < width:
            pattern[y, x] = 1.0
        if 0 <= y2 < height and 0 <= x2 < width:
            pattern[y2, x2] = 1.0
    return cv2.GaussianBlur(pattern, (0, 0), 1.2)


def apply_fft_layer(image: Image.Image, trace_id: str) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    channel = arr[:, :, 2]
    spectrum = np.fft.fftshift(np.fft.fft2(channel))
    pattern = fft_pattern(channel.shape, trace_id)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    magnitude = magnitude * (1.0 + pattern * FFT_DELTA)
    rebuilt = np.real(np.fft.ifft2(np.fft.ifftshift(magnitude * np.exp(1j * phase))))
    arr[:, :, 2] = rebuilt
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def fft_layer_score(image: Image.Image, trace_id: str) -> float:
    channel = np.array(image.convert("RGB"), dtype=np.float32)[:, :, 2]
    magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(channel))))
    pattern = fft_pattern(channel.shape, trace_id)
    mask = pattern > 0.05
    if int(mask.sum()) < 10:
        return 0.0
    selected = magnitude[mask]
    background = magnitude[~mask]
    return float(max(0.0, (selected.mean() - background.mean()) / (background.std() + 1e-6)))


def apply_frequency_layers(
    image: Image.Image,
    trace_id: str,
    *,
    apply_dct_layer_fn: Callable[[Image.Image, str], Image.Image] | None = None,
    apply_dwt_layer_fn: Callable[[Image.Image, str], Image.Image] | None = None,
    apply_fft_layer_fn: Callable[[Image.Image, str], Image.Image] | None = None,
) -> Image.Image:
    dct_fn = apply_dct_layer_fn or apply_dct_layer
    dwt_fn = apply_dwt_layer_fn or apply_dwt_layer
    fft_fn = apply_fft_layer_fn or apply_fft_layer
    return fft_fn(dwt_fn(dct_fn(image, trace_id), trace_id), trace_id)


def layer_scores_for_image(
    image: Image.Image,
    trace_id: str,
    *,
    dct_layer_score_fn: Callable[[Image.Image, str], float] | None = None,
    dwt_layer_score_fn: Callable[[Image.Image, str], float] | None = None,
    fft_layer_score_fn: Callable[[Image.Image, str], float] | None = None,
) -> dict[str, float]:
    dct_fn = dct_layer_score_fn or dct_layer_score
    dwt_fn = dwt_layer_score_fn or dwt_layer_score
    fft_fn = fft_layer_score_fn or fft_layer_score
    return {
        "dct": round(dct_fn(image, trace_id), 4),
        "dwt": round(dwt_fn(image, trace_id), 4),
        "fft": round(fft_fn(image, trace_id), 4),
    }
