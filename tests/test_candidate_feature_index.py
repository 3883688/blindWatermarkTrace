from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from candidate_feature_index import (
    descriptor_match_score,
    extract_feature_descriptors,
    load_feature_descriptors,
    save_feature_descriptors,
)


def feature_image(size: tuple[int, int] = (640, 480), seed: int = 7) -> Image.Image:
    width, height = size
    rng = np.random.default_rng(seed)
    array = rng.integers(20, 235, size=(height, width, 3), dtype=np.uint8)
    array = cv2.GaussianBlur(array, (0, 0), sigmaX=0.8)
    image = Image.fromarray(array, "RGB")
    draw = ImageDraw.Draw(image)
    for index in range(20):
        x = 15 + (index * 71) % (width - 70)
        y = 15 + (index * 43) % (height - 50)
        draw.rectangle((x, y, x + 38, y + 24), outline="white", width=3)
    return image


def test_feature_descriptors_round_trip_through_compressed_index(tmp_path: Path):
    image = feature_image()
    path = tmp_path / "feature.npz"

    descriptors = extract_feature_descriptors(image)
    save_feature_descriptors(path, descriptors)
    restored = load_feature_descriptors(path)

    assert descriptors.dtype == np.uint8
    assert descriptors.ndim == 2
    assert descriptors.shape[1] == 32
    assert np.array_equal(restored, descriptors)


def test_content_match_scores_crop_above_unrelated_image():
    target = feature_image()
    query = target.crop((80, 60, 560, 420)).resize((720, 540), Image.Resampling.BICUBIC)
    unrelated = feature_image(seed=99)

    query_descriptors = extract_feature_descriptors(query)
    target_score = descriptor_match_score(
        query_descriptors,
        extract_feature_descriptors(target),
    )
    unrelated_score = descriptor_match_score(
        query_descriptors,
        extract_feature_descriptors(unrelated),
    )

    assert target_score[0] >= 12
    assert target_score[0] > unrelated_score[0]
    assert target_score[1] > unrelated_score[1]


def test_blank_image_has_no_descriptors_and_no_match():
    blank = Image.new("RGB", (640, 480), "white")
    descriptors = extract_feature_descriptors(blank)

    assert descriptors.shape == (0, 32)
    assert descriptor_match_score(descriptors, descriptors) == (0, 0.0)
