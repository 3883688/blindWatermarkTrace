import { afterEach, expect, test, vi } from 'vitest';
import { createApp, nextTick } from 'vue';
import LoginOverlay from '../src/components/LoginOverlay.vue';

const feedback = vi.hoisted(() => ({ showAlert: vi.fn() }));
vi.mock('../src/ui-feedback.js', () => ({ showAlert: feedback.showAlert }));

let root;

afterEach(() => {
  root?.remove();
  feedback.showAlert.mockReset();
  vi.unstubAllGlobals();
});

test('login failure uses the shared Element Plus feedback', async () => {
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

  await vi.waitFor(() => expect(feedback.showAlert).toHaveBeenCalledWith('用户名或密码错误'));
  await nextTick();
  expect(root.querySelector('.login-error')).toBeNull();
});
