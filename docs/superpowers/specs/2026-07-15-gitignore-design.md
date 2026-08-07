# Git Ignore Rules Design

## Goal

Add a root `.gitignore` that keeps secrets, local caches, logs, and generated runtime data out of version control without excluding source code or distributable releases.

## Rules

The file will ignore:

- `.env`, while `.env.example` remains trackable.
- Python bytecode and cache directories, including `__pycache__/`, `*.py[cod]`, and `.pytest_cache/`.
- The local `.playwright-cli/` directory.
- Runtime log files matching `*.log`.
- Project-generated directories: `output/`, `test_output/`, `uploads/`, and `backups/`.

The file will not ignore `release/`, `assets/`, `data/`, `docs/`, source files, tests, or deployment files.

## Verification

Use `git check-ignore -v` to confirm ignored paths are matched and representative retained paths are not matched. Use `git status --short` to confirm `.env`, caches, logs, and generated directories no longer appear while `.env.example`, `release/`, source files, and documentation remain visible to Git.
