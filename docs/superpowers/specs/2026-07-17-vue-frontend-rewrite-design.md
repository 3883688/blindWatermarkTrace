# Vue Frontend Rewrite Design

## Objective

Replace the current single-file, inline JavaScript frontend with a Vue 3 and Vite application while preserving all existing UI visuals, text, interaction behavior, API contracts, and FastAPI backend business behavior.

## Scope Boundaries

- Do not change Python business logic, FastAPI routes, request fields, response fields, database access, watermark algorithms, authentication rules, or data models.
- The running Vue application must call the existing FastAPI endpoints directly. It must not use local fixture data, a mock API mode, or replacement endpoints.
- Do not add a Node.js runtime requirement to the production service. Node.js is a development/build dependency only.
- Do not change the visual design, page layout, Chinese copy, menu behavior, theme behavior, or user workflow.
- Do not run backend pytest suites for this frontend-only migration.
- Backend changes are limited to static-file packaging support when required for generated frontend assets; no backend business module is changed.

## Architecture

Create a `frontend/` Vite project using Vue 3 and the Composition API. It compiles into `assets/app/`, which is served through the existing FastAPI `/assets` static mount. The root `index.html` becomes a small shell that preserves the current static dependencies and mounts the Vue bundle.

The frontend uses Vue reactive state modules rather than a state-management framework. This avoids an unnecessary dependency while keeping user, role, theme, image list, filters, selection, and modal state separate from view code.

```text
frontend/src/
├── api/             fetch wrapper and endpoint functions
├── state/           reactive user, theme, image, and UI state
├── components/      reusable navigation, upload, dialog, table, filter, and pager parts
├── views/           watermark, trace, image management, role, and user views
├── styles/          tokens, base, layout, component, and page CSS
├── App.vue          page shell and route-like active-view coordination
└── main.js          Vue application bootstrap

assets/app/          Vite production build output, committed for Python-only deployment
index.html           Vue mount shell and compiled asset references
```

## Static Release Support

The existing CentOS release builder recursively packages `assets/` but only allowlists CSS and icon fonts. Add `.js` files to the `assets` suffix allowlist so the compiled Vue bundle is included in the deterministic ZIP. This is a release-packaging change only; it does not affect any backend business behavior.

## API and State Flow

`api/client.js` owns JSON parsing, HTTP failure normalization, and network error normalization. Feature API modules directly call every existing FastAPI endpoint with its current method, path, request payload, and response fields. Vue views invoke state modules; state modules invoke API functions; components receive values and callbacks through props/events.

The local storage keys `currentUser` and `siteTheme` remain unchanged. Existing API paths and response fields are consumed without adaptation layers that alter their meaning.

## Visual Fidelity

Extract the current inline CSS without redesigning it. Preserve CSS variable values, spacing, typography, colors, control dimensions, dialogs, responsive rules, icon classes, labels, and DOM-semantic behavior. Component boundaries may change DOM nesting only where the rendered layout and accessibility behavior remain unchanged.

## Frontend-Only Verification

- Run `npm run build` to prove the production bundle builds.
- Run frontend unit tests against state and API modules with mocked `fetch`, asserting the real endpoint paths, HTTP methods, request fields, and response handling. Mocks are test transport only and are not part of the running application.
- Run browser checks against the built frontend with mocked API responses; do not start or validate FastAPI business logic.
- Capture desktop and mobile screenshots for the login state, watermark view, trace result, management table, and role view. Compare their key layout and visible text against the current UI baseline.
- Run release-builder contract tests only if the static asset allowlist changes. Do not run tests that invoke backend business logic.

## Deployment

The deployment package continues to run `main:app` under the existing `deploy.sh`. The committed `assets/app/` bundle is included in the release ZIP. The server does not run `npm install`, `vite`, or a Node.js process.
