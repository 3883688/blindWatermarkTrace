# Local Tabler Icons Design

## Goal

Remove the page-load dependency on jsDelivr by serving the exact Tabler Icons Webfont `3.44.0` files from the application and including them in the CentOS release package.

## Assets

Store the upstream, unmodified distribution files under:

```text
assets/tabler-icons/tabler-icons.min.css
assets/tabler-icons/fonts/tabler-icons.woff2
assets/tabler-icons/fonts/tabler-icons.woff
assets/tabler-icons/fonts/tabler-icons.ttf
```

The CSS and fonts must come from the pinned npm package version `@tabler/icons-webfont@3.44.0`, not `@latest`. Retain the upstream license header in the CSS. Record SHA-256 values in automated tests so accidental truncation or unreviewed replacement is detected.

## Application Routing

Mount `BASE_DIR / "assets"` at `/assets` through FastAPI `StaticFiles`. The mount is read-only application content and remains separate from mutable `/uploads` data.

Replace the external stylesheet in `index.html` with:

```html
<link rel="stylesheet" href="/assets/tabler-icons/tabler-icons.min.css">
```

No external Tabler, jsDelivr, unpkg, cdnjs, or other icon CDN fallback remains. A missing local asset must be visible as a deployment/package error rather than silently reintroducing network latency.

## Release Packaging

Add the complete `assets/tabler-icons/` directory to `release/trace-v4-centos-20260715.zip`. The package must still exclude `.env`, data, uploads, backups, tests, and the live authentication key.

Rebuild the ZIP, regenerate its `.sha256` file, update the deployment result document, and repeat the clean-extraction HTTP smoke test. The smoke test must verify HTTP 200 for the page, local CSS, and WOFF2 font.

## Tests

- Frontend contract rejects `cdn.jsdelivr.net`, `@latest`, and external Tabler stylesheet URLs.
- Frontend contract requires the local `/assets/tabler-icons/tabler-icons.min.css` URL.
- Asset contract verifies all four files exist, are nonempty, match pinned SHA-256 values, and the CSS references relative local font paths.
- API contract verifies `/assets/tabler-icons/tabler-icons.min.css` returns HTTP 200 with a `text/css` content type and `/assets/tabler-icons/fonts/tabler-icons.woff2` returns HTTP 200 with a `font/woff2` content type.
- CentOS release contract verifies the four asset files are present in the ZIP and no forbidden runtime paths or secrets are added.

## Acceptance Criteria

- Loading the application makes no request to jsDelivr for icons.
- Existing `ti ti-*` classes render without HTML changes.
- Local CSS and WOFF2 respond with HTTP 200.
- Full pytest regression passes.
- The rebuilt CentOS ZIP passes source consistency, secret exclusion, SHA-256, and clean-extraction HTTP smoke checks.
