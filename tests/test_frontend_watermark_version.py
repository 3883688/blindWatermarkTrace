from pathlib import Path


INDEX_HTML = Path(__file__).resolve().parents[1] / "index.html"


def test_embed_result_displays_normalized_watermark_version():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "function watermarkVersionLabel(version)" in html
    assert "水印版本" in html
    assert "watermarkVersionLabel(r.robust_watermark_version)" in html
    assert 'id="robustWatermarkVersion" value="4"' in html
    assert "data.append('robust_watermark_version',document.getElementById('robustWatermarkVersion').value)" in html
    assert "[1,2,3,4].includes(normalized)" in html


def test_frontend_renders_v4_generation_and_recovery_evidence():
    html = INDEX_HTML.read_text(encoding="utf-8")

    for expected in (
        "function isV4Record(r)",
        "function v4GenerationEvidence(r)",
        "function v4RecoveryEvidence(r)",
        "function attributionEvidenceMarkup(r)",
        "认证 DCT",
        "FFT 同步",
        "authenticated_tiles",
        "phase_count",
        "corrected_symbols",
        "bit_errors",
        "sync_confidence",
        "elapsed_ms",
        "认证标签",
    ):
        assert expected in html


def test_frontend_keeps_legacy_scores_out_of_v4_generation_markup():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "function generationEvidenceMarkup(r)" in html
    assert "if(isV4Record(r))return v4GenerationEvidence(r);" in html
    assert "generationEvidenceMarkup(r)" in html
