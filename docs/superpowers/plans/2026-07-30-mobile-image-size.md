# Mobile Layout And Image Size Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the existing mobile navigation/layout changes in the V4-only build and show the generated image size in the original image list.

**Architecture:** Keep the original Vue desktop UI and API contract. Merge only the known responsive navigation, login branding, and CSS changes from the main worktree. Populate the existing `size` response field from the relational `media_objects.byte_size` value for the watermarked output.

**Tech Stack:** Vue 3, Vite, FastAPI, SQLAlchemy, pytest, Vitest, Playwright.

---

### Task 1: Return The Watermarked Image Size

**Files:**
- Modify: `tests/v4/test_original_ui_compat.py`
- Modify: `trace_app/api/compat_v4.py`

- [ ] Add an API regression assertion that `size` is formatted from the output media object's `byte_size`.
- [ ] Run `python -m pytest tests/v4/test_original_ui_compat.py -q` and confirm the assertion fails with the current `"-"` value.
- [ ] Add a small byte-size formatter and read the output media row through the V4 repository.
- [ ] Re-run the focused API test and confirm it passes.

### Task 2: Restore Mobile Navigation And Responsive Layout

**Files:**
- Create: `frontend/tests/mobile-layout-contract.test.js`
- Modify: `frontend/src/components/AppNavigation.vue`
- Modify: `frontend/src/components/LoginOverlay.vue`
- Modify: `frontend/src/styles/index.css`
- Create: `frontend/public/site-logo.png`

- [ ] Add a frontend contract test for the menu toggle, collapsible links, and 560px responsive breakpoint.
- [ ] Run the focused Vitest test and confirm it fails against the current V4 branch.
- [ ] Merge the preserved mobile source changes from the main worktree without copying unrelated frontend edits.
- [ ] Re-run the focused test and build the Vite assets.

### Task 3: Verify And Package

**Files:**
- Modify: `assets/app/app.js`
- Modify: `assets/app/app.css`
- Create: `assets/app/site-logo.png`
- Create: `release/trace-v4-original-ui-relational-<version>.zip`

- [ ] Run the full V4 pytest and frontend Vitest suites.
- [ ] Start the local service with the V4 model path and verify desktop/mobile layouts with browser screenshots.
- [ ] Commit the source and compiled assets.
- [ ] Build a new deployment ZIP, verify its manifest, forbidden-file exclusions, fix markers, and SHA256 checksum.
