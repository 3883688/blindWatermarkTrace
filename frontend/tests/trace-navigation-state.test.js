import { afterEach, expect, test, vi } from 'vitest';
import { createApp, nextTick } from 'vue';
import App from '../src/App.vue';

const okJson = body => new Response(JSON.stringify(body), { status: 200 });
let app;
let root;

afterEach(() => {
  app?.unmount();
  root?.remove();
  app = undefined;
  root = undefined;
  vi.unstubAllGlobals();
  localStorage.clear();
});

test('managed trace result is consumed after its first navigation', async () => {
  localStorage.setItem('currentUser', JSON.stringify({
    username: 'operator',
    role: 'operator',
    menus: ['watermark', 'trace', 'manage'],
    token: 'session',
  }));
  vi.stubGlobal('fetch', vi.fn(path => {
    if (path === '/api/dashboard-stats') {
      return Promise.resolve(okJson({ today: 0, detection_success_rate: 0 }));
    }
    if (path === '/api/images') {
      return Promise.resolve(okJson({
        items: [{
          id: 'managed-record',
          name: 'managed.png',
          download_access_url: '/api/media/managed-image',
          trace_id: 'TR-MANAGED-ONCE',
          robust_watermark_version: 4,
          confidence: 97,
          status: 'V4 认证命中',
        }],
        stats: {},
      }));
    }
    if (path === '/api/watermark/extract-url') {
      return Promise.resolve(okJson({
        trace_id: 'TR-URL-NEW',
        robust_watermark_version: 4,
        confidence: 93,
        status: 'V4 认证命中',
      }));
    }
    throw new Error(`Unexpected request: ${path}`);
  }));
  root = document.createElement('div');
  document.body.append(root);
  app = createApp(App);
  app.config.warnHandler = () => {};
  app.mount(root);

  root.querySelector('[data-menu="manage"]').click();
  await vi.waitFor(() => expect(root.querySelector('button[title="溯源"]')).not.toBeNull());
  root.querySelector('button[title="溯源"]').click();
  await nextTick();
  expect(root.querySelector('.trace-result-card').textContent).toContain('TR-MANAGED-ONCE');
  expect(root.querySelector('.trace-url-box input').value).toBe('/api/media/managed-image');
  expect(root.querySelector('.managed-trace-preview img').getAttribute('src')).toBe('/api/media/managed-image');
  expect(root.querySelector('.managed-trace-preview').textContent).toContain('managed.png');

  root.querySelector('[data-menu="watermark"]').click();
  await nextTick();
  root.querySelector('[data-menu="trace"]').click();
  await nextTick();

  expect(root.querySelector('.trace-result-card').textContent).not.toContain('TR-MANAGED-ONCE');

  const urlInput = root.querySelector('.trace-url-box input');
  urlInput.value = 'https://example.test/new.png';
  urlInput.dispatchEvent(new Event('input'));
  root.querySelector('.url-btn').click();
  await vi.waitFor(() => {
    expect(root.querySelector('.trace-result-card').textContent).toContain('TR-URL-NEW');
  });
});
