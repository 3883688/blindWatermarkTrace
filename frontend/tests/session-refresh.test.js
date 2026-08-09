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

test('reloading the current menu data after logout and login', async () => {
  localStorage.setItem('currentUser', JSON.stringify({
    username: 'operator',
    role: 'operator',
    menus: ['watermark', 'trace', 'manage'],
    token: 'old-session',
  }));
  const fetchMock = vi.fn(path => {
    if (path === '/api/dashboard-stats') {
      return Promise.resolve(okJson({ today: 0, detected_today: 0 }));
    }
    if (path === '/api/images') {
      return Promise.resolve(okJson({ items: [], stats: {} }));
    }
    if (path === '/auth/login') {
      return Promise.resolve(okJson({
        username: 'operator',
        role: 'operator',
        menus: ['watermark', 'trace', 'manage'],
        token: 'new-session',
      }));
    }
    throw new Error(`Unexpected request: ${path}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  root = document.createElement('div');
  document.body.append(root);
  app = createApp(App);
  app.config.warnHandler = () => {};
  app.mount(root);

  root.querySelector('[data-menu="manage"]').click();
  await vi.waitFor(() => {
    expect(fetchMock.mock.calls.filter(([path]) => path === '/api/images')).toHaveLength(1);
  });

  root.querySelector('.btn-logout').click();
  await nextTick();
  const [username, password] = root.querySelectorAll('.login-overlay input');
  username.value = 'operator';
  username.dispatchEvent(new Event('input'));
  password.value = 'secret';
  password.dispatchEvent(new Event('input'));
  root.querySelector('.login-overlay form').dispatchEvent(
    new Event('submit', { bubbles: true, cancelable: true }),
  );

  await vi.waitFor(() => {
    expect(fetchMock.mock.calls.filter(([path]) => path === '/api/images')).toHaveLength(2);
  });
  expect(root.querySelector('[data-menu="manage"]').classList.contains('active')).toBe(true);
});
