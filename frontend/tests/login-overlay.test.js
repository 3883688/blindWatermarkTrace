import { afterEach, expect, test, vi } from 'vitest';
import { createApp, nextTick } from 'vue';
import LoginOverlay from '../src/components/LoginOverlay.vue';

let root;

afterEach(() => {
  root?.remove();
  vi.unstubAllGlobals();
});

test('login failure preserves the legacy alert feedback', async () => {
  const alert = vi.fn();
  vi.stubGlobal('alert', alert);
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
    JSON.stringify({ detail: '用户名或密码错误' }),
    { status: 401 },
  )));
  root = document.createElement('div');
  document.body.append(root);
  createApp(LoginOverlay).mount(root);

  const [username, password] = root.querySelectorAll('input');
  username.value = 'admin';
  username.dispatchEvent(new Event('input'));
  password.value = 'bad';
  password.dispatchEvent(new Event('input'));
  root.querySelector('form').dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));

  await vi.waitFor(() => expect(alert).toHaveBeenCalledWith('用户名或密码错误'));
  await nextTick();
  expect(root.querySelector('.login-error')).toBeNull();
});
