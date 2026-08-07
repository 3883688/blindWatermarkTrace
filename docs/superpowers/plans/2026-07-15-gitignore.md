# Git Ignore Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a root `.gitignore` that excludes secrets, local caches, logs, and selected generated directories while retaining releases and project source assets.

**Architecture:** A single repository-root `.gitignore` defines all rules. Verification uses Git's own `check-ignore` command for positive and negative cases, followed by `git status` to inspect the resulting working tree.

**Tech Stack:** Git ignore patterns, PowerShell

---

### Task 1: Add And Verify Root Ignore Rules

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Verify representative generated paths are not currently ignored**

Run:

```powershell
git check-ignore -- .env __pycache__/main.cpython-313.pyc .pytest_cache/CACHEDIR.TAG .playwright-cli/state.json server.stdout.log output/result.png test_output/report.json uploads/input.png backups/archive.zip
```

Expected: `.pytest_cache/CACHEDIR.TAG` is printed because `.pytest_cache/.gitignore` already ignores its cache contents. The other eight paths are not printed because no repository-root ignore rules exist yet.

- [ ] **Step 2: Create the minimal `.gitignore`**

Create `.gitignore` with exactly:

```gitignore
# Local environment and secrets
.env

# Python bytecode and test caches
__pycache__/
*.py[cod]
.pytest_cache/

# Local browser automation state
.playwright-cli/

# Runtime logs
*.log

# Generated runtime data
output/
test_output/
uploads/
backups/
```

- [ ] **Step 3: Verify paths that must be ignored**

Run:

```powershell
git check-ignore -- .env __pycache__/main.cpython-313.pyc .pytest_cache/CACHEDIR.TAG .playwright-cli/state.json server.stdout.log output/result.png test_output/report.json uploads/input.png backups/archive.zip
```

Expected: exit code 0 and all nine input paths printed.

- [ ] **Step 4: Verify paths that must remain trackable**

Run:

```powershell
git check-ignore --no-index -- .env.example release/app.zip assets/logo.png data/index.json docs/readme.md main.py tests/test_main.py deploy.ps1
```

Expected: exit code 1 with no output because none of the eight paths match an ignore rule.

- [ ] **Step 5: Inspect the working tree**

Run:

```powershell
git status --short
```

Expected: `.env`, `.playwright-cli/`, Python caches, log files, `output/`, `test_output/`, `uploads/`, and `backups/` are absent. `.gitignore`, `.env.example`, `release/`, source code, tests, documentation, assets, and data remain visible as untracked files where applicable.

- [ ] **Step 6: Commit the ignore rules**

Run:

```powershell
git add -- .gitignore
git commit -m "chore: add repository ignore rules"
```

Expected: one commit containing only `.gitignore`.
