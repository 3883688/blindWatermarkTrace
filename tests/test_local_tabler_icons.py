import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

import main


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "assets" / "tabler-icons"
EXPECTED = {
    "tabler-icons.min.css": (
        209066,
        "af452ff18add58c4fdd30aa10d90460aacbf2fb15d07dfd9f2f05d1a1e8d1c48",
    ),
    "fonts/tabler-icons.woff2": (
        457384,
        "bce5d4c933dcfe8708787a3570ab0995a4a99250d6321ed177c7f2179e93eb68",
    ),
    "fonts/tabler-icons.woff": (
        785784,
        "b3ef12f1ddfc007a49c7a51c4cb0f06c3d8f758f60b50ad1a707f09181aa5988",
    ),
    "fonts/tabler-icons.ttf": (
        2810988,
        "bdd463aae8fc706301de6de489ac8429b3ccfbdee8cdf30dc18d647910ff5025",
    ),
}


def test_frontend_uses_only_local_pinned_tabler_stylesheet() -> None:
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    assert (
        '<link rel="stylesheet" '
        'href="/assets/tabler-icons/tabler-icons.min.css">'
    ) in html
    assert "cdn.jsdelivr.net" not in html
    assert "@latest" not in html


def test_pinned_tabler_distribution_files_match_expected_hashes() -> None:
    for relative, (expected_size, expected_hash) in EXPECTED.items():
        path = ASSET_ROOT / relative
        assert path.is_file(), relative
        payload = path.read_bytes()
        canonical_payload = payload.replace(b"\r\n", b"\n") if path.suffix == ".css" else payload
        assert len(canonical_payload) == expected_size, relative
        assert hashlib.sha256(canonical_payload).hexdigest() == expected_hash, relative

    css = (ASSET_ROOT / "tabler-icons.min.css").read_text(encoding="utf-8")
    assert './fonts/tabler-icons.woff2?v3.44.0' in css
    assert './fonts/tabler-icons.woff?' in css
    assert './fonts/tabler-icons.ttf?v3.44.0' in css


def test_fastapi_serves_local_tabler_css_and_woff2() -> None:
    client = TestClient(main.app)

    css = client.get("/assets/tabler-icons/tabler-icons.min.css")
    font = client.get("/assets/tabler-icons/fonts/tabler-icons.woff2")

    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
    assert font.status_code == 200
    assert font.headers["content-type"].startswith("font/woff2")
