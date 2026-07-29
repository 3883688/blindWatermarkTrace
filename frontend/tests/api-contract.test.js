import { afterEach, describe, expect, test, vi } from 'vitest';
import {
  createUser,
  deleteImage,
  embedWatermark,
  extractUpload,
  extractUrl,
  listImages,
  listRoles,
  listUsers,
  login,
  saveRole,
  saveUser,
} from '../src/api/trace.js';
import { request } from '../src/api/client.js';
import { createAppState } from '../src/state/app.js';
import { createImageState } from '../src/state/images.js';

const okJson = body => new Response(JSON.stringify(body), { status: 200 });

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

test('login posts the existing form endpoint and fields', async () => {
  const fetchMock = vi.fn().mockResolvedValue(okJson({ username: 'admin' }));
  vi.stubGlobal('fetch', fetchMock);

  await login('admin', 'secret');

  expect(fetchMock).toHaveBeenCalledWith('/auth/login', expect.objectContaining({ method: 'POST' }));
  const body = fetchMock.mock.calls[0][1].body;
  expect(body).toBeInstanceOf(FormData);
  expect([...body.entries()]).toEqual([['username', 'admin'], ['password', 'secret']]);
});

test('authenticated requests send the cached login token', async () => {
  localStorage.setItem('currentUser', JSON.stringify({ token: 'local-session' }));
  const fetchMock = vi.fn().mockResolvedValue(okJson({ items: [] }));
  vi.stubGlobal('fetch', fetchMock);

  await listImages();

  expect(fetchMock.mock.calls[0][1].headers.Authorization).toBe('Bearer local-session');
});

test('login does not send a stale cached token', async () => {
  localStorage.setItem('currentUser', JSON.stringify({ token: 'stale' }));
  const fetchMock = vi.fn().mockResolvedValue(okJson({ token: 'fresh' }));
  vi.stubGlobal('fetch', fetchMock);

  await login('admin', 'secret');

  expect(fetchMock.mock.calls[0][1].headers.Authorization).toBeUndefined();
});

test('authenticated 401 responses clear cached login state', async () => {
  localStorage.setItem('currentUser', JSON.stringify({ token: 'expired' }));
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
    JSON.stringify({ detail: '登录已失效，请重新登录' }),
    { status: 401 },
  )));

  await expect(listImages()).rejects.toThrow('登录已失效，请重新登录');

  expect(localStorage.getItem('currentUser')).toBeNull();
});

test('watermark requests use V4-only endpoint paths and payloads', async () => {
  const fetchMock = vi.fn().mockImplementation(() => okJson({}));
  vi.stubGlobal('fetch', fetchMock);
  const upload = new FormData();
  const url = new FormData();
  url.append('url', 'https://example.test/image.png');

  await embedWatermark(upload);
  await extractUpload(upload);
  await extractUrl(url);

  expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
    '/api/v4/generate',
    '/api/v4/detect',
    '/api/v4/detect-url',
  ]);
  expect(fetchMock.mock.calls.every(([, options]) => options.method === 'POST')).toBe(true);
  expect(fetchMock.mock.calls[2][1].body).toBe(url);
});

test('management requests use V4 records and current administration paths', async () => {
  const fetchMock = vi.fn().mockImplementation(() => okJson({}));
  vi.stubGlobal('fetch', fetchMock);

  await listImages();
  await deleteImage('image/a');
  await listRoles();
  await saveRole('operator', ['trace']);
  await listUsers();
  await createUser({ username: 'new-user', password: 'secret', role: 'operator' });
  await saveUser('new/user', 'admin');

  expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
    '/api/v4/records',
    '/api/v4/records/image%2Fa',
    '/api/roles',
    '/api/roles/operator',
    '/api/users',
    '/api/users',
    '/api/users/new%2Fuser',
  ]);
  expect(fetchMock.mock.calls[3][1]).toMatchObject({ method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ menus: ['trace'] }) });
  expect(fetchMock.mock.calls[5][1].body).toBe(JSON.stringify({ username: 'new-user', password: 'secret', role: 'operator' }));
  expect(fetchMock.mock.calls[6][1].body).toBe(JSON.stringify({ role: 'admin' }));
});

describe('request', () => {
  test('normalizes non-OK response details', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(okJson({ detail: 'forbidden' })));
    globalThis.fetch.mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'forbidden' }), { status: 403 }));
    await expect(request('/api/images')).rejects.toThrow('forbidden');
  });

  test('normalizes malformed JSON from successful responses', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('not json', { status: 200 })));
    await expect(request('/api/images')).rejects.toThrow('响应格式无效');
  });

  test('normalizes malformed JSON from failed responses', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('gateway error', { status: 502 })));
    await expect(request('/api/images')).rejects.toThrow('请求失败 (502)');
  });
});

test('app state preserves currentUser and siteTheme storage keys', () => {
  localStorage.setItem('siteTheme', 'light');
  localStorage.setItem('currentUser', JSON.stringify({ username: 'operator', role: 'operator', menus: ['trace'] }));
  const state = createAppState();

  expect(state.theme).toBe('light');
  expect(state.currentUser.username).toBe('operator');
  expect(state.visibleMenus).toEqual(['trace']);
});

test('image state filters and paginates records for the management view', () => {
  const state = createImageState([
    { id: '1', status: '保护中', created_at: '2026-07-17' },
    { id: '2', status: '已删除', created_at: '2026-07-16' },
  ], 1);
  state.activeFilter = '保护中';

  expect(state.filteredRows.value.map(item => item.id)).toEqual(['1']);
  expect(state.pagedRows.value.map(item => item.id)).toEqual(['1']);
  expect(state.totalPages.value).toBe(1);
});
