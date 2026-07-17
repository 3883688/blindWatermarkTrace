# Vue Frontend Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the existing single-file frontend as a Vue 3 application without changing visuals, user workflows, FastAPI business behavior, or real API contracts.

**Architecture:** Vue source lives under `frontend/` and builds to committed `assets/app/` static files. Components use state modules and direct API modules; the API modules call the existing FastAPI endpoints exactly. The root HTML remains a static FastAPI-served shell, while the release builder includes compiled JavaScript alongside existing CSS and fonts.

**Tech Stack:** Vue 3, Vite, Vitest, Playwright, native `fetch`, existing FastAPI static file mount

---

## File Structure

- Create: `frontend/package.json` - Node development scripts and pinned frontend dependencies.
- Create: `frontend/vite.config.js` - development server proxy and deterministic production output.
- Create: `frontend/src/main.js` - Vue bootstrap.
- Create: `frontend/src/App.vue` - shell, active-view selection, authentication overlay coordination.
- Create: `frontend/src/api/client.js` - direct fetch client with normalized HTTP failures.
- Create: `frontend/src/api/trace.js` - exact existing FastAPI endpoint functions.
- Create: `frontend/src/state/app.js` - reactive user/theme/page state using existing local-storage keys.
- Create: `frontend/src/state/images.js` - reactive records, filters, pagination, selection, export state.
- Create: `frontend/src/components/` - navigation, dialogs, upload controls, result markup, table, filters, pager.
- Create: `frontend/src/views/` - watermark, trace, management, role, and user views.
- Create: `frontend/src/styles/` - extracted tokens, layout, component, and page CSS.
- Create: `frontend/tests/` - Vitest API/state tests and Playwright visual flows with transport-only mocks.
- Modify: `index.html` - preserve static shell metadata and load `/assets/app/app.js` into `#app`.
- Modify: `tools/build_centos_release.py` - allow `.js` within the `assets/` release tree.
- Modify: `tests/test_release_builder.py` - assert compiled asset JavaScript is admitted to the release package.
- Generate: `assets/app/app.js`, `assets/app/app.css`, and required Vite chunks - committed production frontend bundle.
- Regenerate: `release/trace-v4-centos-20260715/` and its ZIP/checksum through the existing release builder.

### Task 1: Vite Build Foundation

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.js`
- Create: `frontend/src/main.js`
- Create: `frontend/src/App.vue`
- Modify: `index.html`

- [ ] **Step 1: Write a failing frontend build contract**

Create `frontend/tests/build-contract.test.js`:

```js
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { expect, test } from 'vitest';

test('production bundle has a stable FastAPI asset entry', () => {
  expect(existsSync(fileURLToPath(new URL('../../assets/app/app.js', import.meta.url)))).toBe(true);
});
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `npm --prefix frontend test -- --run frontend/tests/build-contract.test.js`

Expected: FAIL because the `frontend` package and compiled entry do not exist.

- [ ] **Step 3: Create the Vite configuration and Vue entry**

Create `frontend/package.json`:

```json
{
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:ui": "playwright test"
  },
  "dependencies": { "vue": "^3.5.0" },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.2.0",
    "@playwright/test": "^1.51.0",
    "jsdom": "^26.0.0",
    "vite": "^6.1.0",
    "vitest": "^3.0.0"
  }
}
```

Create `frontend/vite.config.js`:

```js
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  base: '/assets/app/',
  plugins: [vue()],
  build: {
    outDir: '../assets/app',
    emptyOutDir: true,
    assetsDir: '',
    rollupOptions: {
      output: {
        entryFileNames: 'app.js',
        chunkFileNames: 'chunks/[name]-[hash].js',
        assetFileNames: 'app.[ext]'
      }
    }
  },
  server: { proxy: { '/api': 'http://127.0.0.1:8000', '/auth': 'http://127.0.0.1:8000' } },
  test: { environment: 'jsdom' }
});
```

Create `frontend/src/main.js`:

```js
import { createApp } from 'vue';
import App from './App.vue';
import './styles/index.css';

createApp(App).mount('#app');
```

Create `frontend/src/App.vue` with the existing application shell placeholders:

```vue
<template>
  <main id="vue-app" data-theme="dark"><h1 class="sr-only">WatermarkSystem</h1></main>
</template>
```

Replace the current body content and inline style/script in `index.html` with:

```html
<body>
  <div id="app"></div>
  <script type="module" src="/assets/app/app.js"></script>
</body>
```

- [ ] **Step 4: Install and build the frontend**

Run: `npm --prefix frontend install` then `npm --prefix frontend run build`

Expected: `assets/app/app.js` and `assets/app/app.css` exist.

- [ ] **Step 5: Run the build contract test**

Run: `npm --prefix frontend test -- --run frontend/tests/build-contract.test.js`

Expected: PASS.

- [ ] **Step 6: Commit the frontend build foundation**

```powershell
git add -- frontend/package.json frontend/package-lock.json frontend/vite.config.js frontend/src/main.js frontend/src/App.vue frontend/tests/build-contract.test.js index.html assets/app
git commit -m "feat: add Vue frontend build foundation"
```

### Task 2: Real API Client and Shared State

**Files:**
- Create: `frontend/src/api/client.js`
- Create: `frontend/src/api/trace.js`
- Create: `frontend/src/state/app.js`
- Create: `frontend/src/state/images.js`
- Create: `frontend/tests/api-contract.test.js`

- [ ] **Step 1: Write failing real-endpoint contract tests**

Create `frontend/tests/api-contract.test.js`:

```js
import { afterEach, expect, test, vi } from 'vitest';
import { embedWatermark, extractUpload, login } from '../src/api/trace.js';

afterEach(() => vi.unstubAllGlobals());

test('login posts the existing form endpoint', async () => {
  const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ username: 'admin' }), { status: 200 }));
  vi.stubGlobal('fetch', fetchMock);
  await login('admin', 'secret');
  expect(fetchMock).toHaveBeenCalledWith('/auth/login', expect.objectContaining({ method: 'POST' }));
});

test('watermark requests retain the existing endpoint paths', async () => {
  const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({}), { status: 200 }));
  vi.stubGlobal('fetch', fetchMock);
  await embedWatermark(new FormData());
  await extractUpload(new FormData());
  expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
    '/api/watermark/embed',
    '/api/watermark/extract'
  ]);
});
```

- [ ] **Step 2: Run API tests and verify they fail**

Run: `npm --prefix frontend test -- --run frontend/tests/api-contract.test.js`

Expected: FAIL because `trace.js` does not exist.

- [ ] **Step 3: Implement direct API functions and reactive state**

Create `frontend/src/api/client.js`:

```js
export async function request(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `请求失败 (${response.status})`);
  return body;
}
```

Create `frontend/src/api/trace.js` with direct existing routes:

```js
import { request } from './client.js';

export const login = (username, password) => {
  const body = new FormData();
  body.append('username', username);
  body.append('password', password);
  return request('/auth/login', { method: 'POST', body });
};
export const embedWatermark = body => request('/api/watermark/embed', { method: 'POST', body });
export const extractUpload = body => request('/api/watermark/extract', { method: 'POST', body });
export const extractUrl = body => request('/api/watermark/extract-url', { method: 'POST', body });
export const listImages = () => request('/api/images');
export const deleteImage = id => request(`/api/images/${encodeURIComponent(id)}`, { method: 'DELETE' });
export const listRoles = () => request('/api/roles');
export const saveRole = (key, menus) => request(`/api/roles/${encodeURIComponent(key)}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ menus }) });
export const listUsers = () => request('/api/users');
export const createUser = payload => request('/api/users', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
export const saveUser = (username, role) => request(`/api/users/${encodeURIComponent(username)}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ role }) });
export const dashboardStats = () => request('/api/dashboard-stats');
```

Create `frontend/src/state/app.js` with `currentUser`, `siteTheme`, `activePage`, and the unchanged `localStorage` keys. Create `frontend/src/state/images.js` with `records`, `activeFilter`, `currentPage`, `selectedIds`, `filteredRows`, and `pagedRows` as Vue `reactive`/`computed` values.

- [ ] **Step 4: Run API tests**

Run: `npm --prefix frontend test -- --run frontend/tests/api-contract.test.js`

Expected: PASS without starting FastAPI or changing backend code.

- [ ] **Step 5: Commit API and state modules**

```powershell
git add -- frontend/src/api frontend/src/state frontend/tests/api-contract.test.js
git commit -m "feat: add Vue API and state modules"
```

### Task 3: Preserve Shell, Theme, Login, and Navigation

**Files:**
- Create: `frontend/src/components/AppNavigation.vue`
- Create: `frontend/src/components/LoginOverlay.vue`
- Create: `frontend/src/styles/index.css`
- Create: `frontend/src/styles/tokens.css`
- Create: `frontend/src/styles/layout.css`
- Modify: `frontend/src/App.vue`
- Test: `frontend/tests/app-state.test.js`

- [ ] **Step 1: Write failing state-preservation test**

Create `frontend/tests/app-state.test.js`:

```js
import { expect, test } from 'vitest';
import { createAppState } from '../src/state/app.js';

test('existing local storage keys preserve theme and current user', () => {
  localStorage.setItem('siteTheme', 'light');
  localStorage.setItem('currentUser', JSON.stringify({ username: 'operator', role: 'operator', menus: ['trace'] }));
  const state = createAppState();
  expect(state.theme).toBe('light');
  expect(state.currentUser.username).toBe('operator');
  expect(state.visibleMenus).toEqual(['trace']);
});
```

- [ ] **Step 2: Run the state test and verify it fails**

Run: `npm --prefix frontend test -- --run frontend/tests/app-state.test.js`

Expected: FAIL because `createAppState` is absent.

- [ ] **Step 3: Extract existing visual CSS and implement shell components**

Move current CSS declarations from `index.html` into `tokens.css`, `layout.css`, and `index.css` without changing selectors or values. Implement `createAppState` so it reads/writes `siteTheme` and `currentUser`, exposes `visibleMenus`, and lets `AppNavigation` emit the active page. Implement `LoginOverlay` using `login()` and the existing visible labels.

Use this root composition in `App.vue`:

```vue
<template>
  <div class="app-shell" :data-theme="state.theme">
    <AppNavigation :menus="state.visibleMenus" :active-page="state.activePage" @navigate="state.activePage = $event" />
    <LoginOverlay v-if="!state.currentUser" @logged-in="state.setUser" />
    <WatermarkView v-if="state.activePage === 'watermark'" />
    <TraceView v-else-if="state.activePage === 'trace'" />
    <ManageView v-else-if="state.activePage === 'manage'" />
    <RoleView v-else />
  </div>
</template>
```

- [ ] **Step 4: Run the state test and production build**

Run: `npm --prefix frontend test -- --run frontend/tests/app-state.test.js` and `npm --prefix frontend run build`

Expected: both PASS.

- [ ] **Step 5: Commit shell migration**

```powershell
git add -- frontend/src/App.vue frontend/src/components/AppNavigation.vue frontend/src/components/LoginOverlay.vue frontend/src/styles frontend/src/state/app.js frontend/tests/app-state.test.js assets/app
git commit -m "feat: migrate shell and authentication to Vue"
```

### Task 4: Migrate Watermark and Trace Workflows

**Files:**
- Create: `frontend/src/components/FileDropzone.vue`
- Create: `frontend/src/components/ResultDialog.vue`
- Create: `frontend/src/views/WatermarkView.vue`
- Create: `frontend/src/views/TraceView.vue`
- Test: `frontend/tests/watermark-form.test.js`

- [ ] **Step 1: Write failing form-payload test**

Create `frontend/tests/watermark-form.test.js`:

```js
import { expect, test } from 'vitest';
import { watermarkFormData } from '../src/views/watermark-form.js';

test('watermark form retains current backend field names', () => {
  const data = watermarkFormData(new File(['image'], 'source.png'), { userId: 'operator', robustVersion: '4' });
  expect([...data.keys()]).toEqual(expect.arrayContaining([
    'file', 'user_id', 'mode', 'copyright_enabled', 'robust_watermark_version',
    'small_crop_trace_enabled', 'dot_matrix_trace_enabled'
  ]));
});
```

- [ ] **Step 2: Run the form test and verify it fails**

Run: `npm --prefix frontend test -- --run frontend/tests/watermark-form.test.js`

Expected: FAIL because `watermark-form.js` is absent.

- [ ] **Step 3: Implement form composition and views**

Create `watermark-form.js` to append the exact existing `FormData` field names and defaults. `WatermarkView` must preserve current labels, controls, preview, range values, dialog result fields, and links. `TraceView` must preserve both file and URL tracing, clear the inactive source, use `extractUpload`/`extractUrl`, and render current evidence fields including V4 recovery evidence.

Use `FileDropzone` props `inputId`, `accept`, and `modelValue`; emit the selected `File`. Revoke object URLs in `onBeforeUnmount`. Use `ResultDialog` around native `<dialog>` and retain close-on-backdrop behavior.

- [ ] **Step 4: Run unit tests and build**

Run: `npm --prefix frontend test -- --run frontend/tests/watermark-form.test.js` and `npm --prefix frontend run build`

Expected: both PASS.

- [ ] **Step 5: Commit watermark and trace views**

```powershell
git add -- frontend/src/components/FileDropzone.vue frontend/src/components/ResultDialog.vue frontend/src/views frontend/tests/watermark-form.test.js assets/app
git commit -m "feat: migrate watermark and trace views to Vue"
```

### Task 5: Migrate Management, Roles, and Users

**Files:**
- Create: `frontend/src/components/ImageTable.vue`
- Create: `frontend/src/components/ImageFilters.vue`
- Create: `frontend/src/components/Pager.vue`
- Create: `frontend/src/views/ManageView.vue`
- Create: `frontend/src/views/RoleView.vue`
- Create: `frontend/src/views/UserView.vue`
- Test: `frontend/tests/image-state.test.js`

- [ ] **Step 1: Write failing filter and pagination test**

Create `frontend/tests/image-state.test.js`:

```js
import { expect, test } from 'vitest';
import { createImageState } from '../src/state/images.js';

test('management state preserves filter and page behavior', () => {
  const state = createImageState([{ id: '1', status: '保护中', created_at: '2026-07-17' }]);
  state.activeFilter = '保护中';
  expect(state.filteredRows.value).toHaveLength(1);
  expect(state.pagedRows.value.map(item => item.id)).toEqual(['1']);
});
```

- [ ] **Step 2: Run the state test and verify it fails**

Run: `npm --prefix frontend test -- --run frontend/tests/image-state.test.js`

Expected: FAIL because `createImageState` is absent.

- [ ] **Step 3: Implement management components and direct endpoint behavior**

Implement `ManageView` with current filter labels, search, sorting, selection, CSV export, preview, delete confirmation, and status/count rendering. `ImageTable` must escape all visible record text via Vue interpolation and use `:href`/`:src` bindings. `RoleView` must use `listRoles`/`saveRole`; `UserView` must use `listUsers`/`createUser`/`saveUser`. Preserve current admin-only visibility and labels.

- [ ] **Step 4: Run state test and build**

Run: `npm --prefix frontend test -- --run frontend/tests/image-state.test.js` and `npm --prefix frontend run build`

Expected: both PASS.

- [ ] **Step 5: Commit management migration**

```powershell
git add -- frontend/src/components/ImageTable.vue frontend/src/components/ImageFilters.vue frontend/src/components/Pager.vue frontend/src/views/ManageView.vue frontend/src/views/RoleView.vue frontend/src/views/UserView.vue frontend/src/state/images.js frontend/tests/image-state.test.js assets/app
git commit -m "feat: migrate management views to Vue"
```

### Task 6: Frontend Visual Verification and Release Packaging

**Files:**
- Create: `frontend/playwright.config.js`
- Create: `frontend/tests/ui/app.spec.js`
- Modify: `tools/build_centos_release.py`
- Modify: `tests/test_release_builder.py`
- Regenerate: `release/trace-v4-centos-20260715/`
- Regenerate: `release/trace-v4-centos-20260715.zip`
- Regenerate: `release/trace-v4-centos-20260715.zip.sha256`

- [ ] **Step 1: Write a failing release asset test**

Add to `tests/test_release_builder.py`:

```python
def test_release_source_filter_allows_compiled_frontend_javascript() -> None:
    assert is_release_source(Path("assets/app/app.js"))
```

- [ ] **Step 2: Run the release test and verify it fails**

Run: `python -m pytest tests/test_release_builder.py::test_release_source_filter_allows_compiled_frontend_javascript -q`

Expected: FAIL because `.js` is excluded from the assets allowlist.

- [ ] **Step 3: Include compiled JavaScript in the release allowlist**

Change the assets allowlist in `tools/build_centos_release.py` to:

```python
RECURSIVE_SUFFIX_ALLOWLIST = {
    "assets": {".css", ".js", ".ttf", ".woff", ".woff2"},
    "trace_app": {".py"},
    "watermark_v4": {".py"},
}
```

- [ ] **Step 4: Add frontend-only browser verification**

Create `frontend/tests/ui/app.spec.js` with a route stub that returns current API response shapes and never sends a request to FastAPI:

```js
import { expect, test } from '@playwright/test';

test('Vue shell retains core workflow labels at desktop and mobile', async ({ page }) => {
  await page.route('**/api/**', route => route.fulfill({ contentType: 'application/json', body: JSON.stringify({ items: [], stats: {}, roles: {}, menus: {}, users: {} }) }));
  await page.route('**/auth/login', route => route.fulfill({ contentType: 'application/json', body: JSON.stringify({ username: 'admin', role: 'admin', menus: ['watermark', 'trace', 'manage', 'role'] }) }));
  await page.goto('http://127.0.0.1:4173');
  await expect(page.getByText('生成水印')).toBeVisible();
  await expect(page.getByText('图片溯源')).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByText('生成水印')).toBeVisible();
});
```

Configure Playwright `webServer` to run `npm run preview -- --host 127.0.0.1 --port 4173` from `frontend/`. Capture screenshots for login, watermark, trace result, management, and role states in `frontend/test-results/`; do not commit test results.

- [ ] **Step 5: Run frontend-only verification and release contracts**

Run:

```powershell
npm --prefix frontend test -- --run
npm --prefix frontend run build
npm --prefix frontend run test:ui
python -m pytest tests/test_release_builder.py tests/test_centos_deploy_contract.py -q
```

Expected: frontend tests/build/browser checks PASS; release tests PASS; no Python watermark/auth/database business tests run.

- [ ] **Step 6: Rebuild and verify the CentOS ZIP**

Run: `python tools/build_centos_release.py`

Run:

```powershell
$archive = 'release/trace-v4-centos-20260715.zip'
$actual = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLower()
$recorded = ((Get-Content "$archive.sha256" -Raw).Split()[0]).ToLower()
if ($actual -ne $recorded) { throw 'release checksum mismatch' }
```

Expected: builder prints a digest and the ZIP checksum matches its `.sha256` file.

- [ ] **Step 7: Commit the complete Vue migration and release artifacts**

```powershell
git add -- frontend index.html assets/app tools/build_centos_release.py tests/test_release_builder.py release/trace-v4-centos-20260715 release/trace-v4-centos-20260715.zip release/trace-v4-centos-20260715.zip.sha256
git commit -m "feat: rewrite frontend with Vue"
```
