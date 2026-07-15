from pathlib import Path

import numpy as np
from PIL import Image

import main


ROOT = Path(__file__).resolve().parents[1]


def test_prominent_corner_label_adds_yellow_text_to_bottom_right():
    source = Image.new("RGB", (800, 500), (80, 90, 100))

    result = main.draw_prominent_corner_label(source, "Copyright Example")
    pixels = np.asarray(result)
    corner = pixels[300:, 350:]

    yellow = (corner[:, :, 0] > 220) & (corner[:, :, 1] > 170) & (corner[:, :, 2] < 80)
    dark = (corner[:, :, 0] < 40) & (corner[:, :, 1] < 40) & (corner[:, :, 2] < 40)
    assert int(yellow.sum()) > 100
    assert int(dark.sum()) > 100


def test_frontend_submits_prominent_corner_option():
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    assert 'id="copyrightProminentCornerEnabled"' in html
    assert "copyright_prominent_corner_enabled" in html

