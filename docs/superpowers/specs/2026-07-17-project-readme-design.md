# Project README Design

## Objective

Create a root `README.md` that introduces Trace System to open-source readers while providing accurate local-development guidance and a clear path to the existing production deployment guide.

## Audience

- Developers who need to run, inspect, test, or extend the FastAPI application.
- Operators who need the production deployment entry point without duplicating its CentOS/MySQL instructions.
- Evaluators who need a concise overview of the watermark and image-tracing capabilities.

## Content Structure

1. Project title, concise purpose statement, and `demo/1.jpg` as the repository-local demo image.
2. Core capabilities: watermark generation, multi-algorithm detection, evidence records, authentication/roles, and image management.
3. Exact-file tracing behavior: MD5 selects candidate records, SHA-256 confirms attribution, original and watermarked exact matches return immediately, and historical SHA-256-only records remain supported.
4. Modular architecture diagram and directory responsibilities, reflecting the actual `main.py` compatibility entry and `trace_app` application tree.
5. Local prerequisites, dependency installation, environment configuration, SQLite development example, Uvicorn startup command, and local URL.
6. Selected API endpoints grouped by auth, watermark, image, user/role, and dashboard responsibilities.
7. Focused test commands and a link to `README_DEPLOY.md` for the authoritative CentOS deployment instructions.
8. Security and repository-data boundaries: keep secrets in `.env`; do not commit databases, uploads, or generated indexes.

## Boundaries

- `README.md` is written in Chinese, matching the existing deployment guide and project-facing interface.
- The README links to `README_DEPLOY.md`; it does not copy deployment steps that would become a second source of truth.
- The README references the existing `demo/1.jpg`; it does not generate, transform, rename, or embed a new image asset.
- The change is documentation-only. It does not alter application behavior, APIs, dependencies, release artifacts, or tests.

## Validation

- Confirm every referenced local path exists.
- Check Markdown headings, fenced commands, and image path syntax.
- Confirm `git diff --check` succeeds.
- Commit only `README.md` and this design/plan documentation; leave untracked runtime data and user images untouched.
