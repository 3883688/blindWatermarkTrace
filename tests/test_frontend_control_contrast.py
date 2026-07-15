from pathlib import Path


INDEX_HTML = Path(__file__).resolve().parents[1] / "index.html"


def test_result_pagination_and_action_controls_have_high_contrast_styles():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert ".result-link:hover" in html
    assert ".result-link:focus-visible" in html
    assert ".page-btn:focus-visible" in html
    assert ".icon-btn:focus-visible" in html
    assert '[data-theme="light"] .result-link' in html
    assert '[data-theme="light"] .icon-btn' in html

