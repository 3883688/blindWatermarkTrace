from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

import main


INDEX_HTML = Path(__file__).resolve().parents[1] / "index.html"


def test_dashboard_stats_excludes_image_records(monkeypatch):
    monkeypatch.setattr(
        main,
        "read_records",
        lambda: [
            {
                "id": "today",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            {"id": "old", "created_at": "2020-01-01 00:00:00"},
        ],
    )
    monkeypatch.setattr(
        main,
        "read_detection_stats",
        lambda: {"attempts": 4, "successes": 3},
    )

    response = TestClient(main.app).get("/api/dashboard-stats")

    assert response.status_code == 200
    assert response.json() == {"today": 1, "detection_success_rate": 75.0}


def test_homepage_initialization_does_not_load_image_records():
    html = INDEX_HTML.read_text(encoding="utf-8")
    initialization = html.rsplit("setupDropzone('dz1','fi1');", 1)[1]

    assert "async function loadDashboardStats()" in html
    assert "if(p==='manage')loadImages();" in html
    assert "loadDashboardStats();" in initialization
    assert "loadImages();" not in initialization
