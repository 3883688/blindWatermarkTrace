# Homepage Data Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent image records and thumbnails from loading on the homepage while preserving immediately visible homepage statistics.

**Architecture:** Add a dedicated FastAPI endpoint that returns only dashboard statistics. Initialize the frontend with that endpoint, while retaining the existing image-list request behind the image-management navigation path and record-refresh operations.

**Tech Stack:** Python, FastAPI, pytest, browser JavaScript in a single HTML file

---

### Task 1: Add failing dashboard-loading regression tests

**Files:**
- Create: `tests/test_homepage_data_loading.py`
- Test: `tests/test_homepage_data_loading.py`

- [ ] **Step 1: Write the failing endpoint contract test**

```python
from pathlib import Path

from fastapi.testclient import TestClient

import main


INDEX_HTML = Path(__file__).resolve().parents[1] / "index.html"


def test_dashboard_stats_excludes_image_records(monkeypatch):
    monkeypatch.setattr(
        main,
        "read_records",
        lambda: [
            {"id": "today", "created_at": main.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
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
```

- [ ] **Step 2: Write the failing frontend initialization test**

```python
def test_homepage_initialization_does_not_load_image_records():
    html = INDEX_HTML.read_text(encoding="utf-8")
    initialization = html.rsplit("setupDropzone('dz1','fi1');", 1)[1]

    assert "async function loadDashboardStats()" in html
    assert "if(p==='manage')loadImages();" in html
    assert "loadDashboardStats();" in initialization
    assert "loadImages();" not in initialization
```

- [ ] **Step 3: Run tests and verify RED**

Run: `pytest tests/test_homepage_data_loading.py -q`

Expected: both tests fail because `/api/dashboard-stats` and `loadDashboardStats()` do not exist, and initialization still calls `loadImages()`.

### Task 2: Add the lightweight dashboard endpoint

**Files:**
- Modify: `main.py:4063`
- Test: `tests/test_homepage_data_loading.py`

- [ ] **Step 1: Implement the endpoint with the required response only**

Insert before the existing `/api/images` endpoint:

```python
@app.get("/api/dashboard-stats")
def dashboard_stats() -> dict[str, int | float]:
    records = read_records()
    detection_stats = read_detection_stats()
    attempts = detection_stats["attempts"]
    successes = detection_stats["successes"]
    success_rate = round((successes / attempts) * 100, 1) if attempts else 0.0
    return {
        "today": today_watermark_count(records),
        "detection_success_rate": success_rate,
    }
```

- [ ] **Step 2: Run the endpoint test and verify GREEN**

Run: `pytest tests/test_homepage_data_loading.py::test_dashboard_stats_excludes_image_records -q`

Expected: `1 passed`.

### Task 3: Defer image loading in the frontend

**Files:**
- Modify: `index.html:1200-1276`
- Test: `tests/test_homepage_data_loading.py`

- [ ] **Step 1: Add the lightweight frontend loader**

Place it next to `updateDashboardStats`:

```javascript
async function loadDashboardStats(){
  try{
    const res=await fetch('/api/dashboard-stats');
    if(!res.ok)return;
    updateDashboardStats({stats:await res.json()});
  }catch(_err){}
}
```

- [ ] **Step 2: Replace the initialization image load**

Replace the final call:

```javascript
loadImages();
```

with:

```javascript
loadDashboardStats();
```

Keep `if(p==='manage')loadImages();` unchanged so image rows and thumbnails load when image management opens.

- [ ] **Step 3: Run the focused tests and verify GREEN**

Run: `pytest tests/test_homepage_data_loading.py -q`

Expected: `2 passed`.

- [ ] **Step 4: Run lightweight related frontend tests**

Run: `pytest tests/test_homepage_data_loading.py tests/test_frontend_control_contrast.py tests/test_frontend_watermark_version.py tests/test_local_tabler_icons.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Verify the modified frontend call sites**

Run: `rg -n "loadDashboardStats|loadImages\(\)" index.html`

Expected: initialization calls only `loadDashboardStats()`, while `loadImages()` remains in image-management navigation and explicit record-refresh flows.

### Task 4: Version-control checkpoint

**Files:**
- Modify: none

- [ ] **Step 1: Record repository limitation**

The workspace root has no `.git` metadata, so no commit command can be run. Do not initialize a repository as part of this fix.
