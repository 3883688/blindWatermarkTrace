import { expect, test } from '@playwright/test';

const admin = {
  username: 'admin',
  role: 'admin',
  menus: ['watermark', 'trace', 'manage', 'role'],
};
const previewPng = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  'base64',
);

function apiResponse(url) {
  if (url.pathname === '/api/dashboard-stats') {
    return { today: 3, detected_today: 12 };
  }
  if (url.pathname === '/api/images') {
    return {
      items: [{
        id: 'record-1', name: 'sample.jpg', user_id: 'admin', trace_id: 'trace-1',
        size: '2.5 MB', mode: 'dct', created_at: '2026-07-17', status: '保护中', confidence: 98,
        robust_watermark_version: 4, download_access_url: '/api/media/record-1',
      }],
      stats: { total: 1, protected: 1, leaks: 0, hits: 1 },
    };
  }
  if (url.pathname === '/api/roles') {
    return {
      roles: { admin: { label: '管理员', menus: admin.menus } },
      menus: { watermark: '生成水印', trace: '图片溯源', manage: '图片管理', role: '角色管理' },
    };
  }
  if (url.pathname === '/api/users') {
    return { users: { admin: { role: 'admin' } } };
  }
  if (url.pathname === '/api/watermark/extract-url') {
    return {
      user_id: 'admin', trace_id: 'trace-1', mode_label: 'DCT + 空间域',
      created_at: '2026-07-17', confidence: 98, phash_match: true, status: '保护中',
    };
  }
  return {};
}

async function installRouteMocks(page) {
  await page.route('**/auth/**', route => route.fulfill({
    contentType: 'application/json', body: JSON.stringify(admin),
  }));
  await page.route('**/api/**', route => route.fulfill({
    contentType: 'application/json', body: JSON.stringify(apiResponse(new URL(route.request().url()))),
  }));
  await page.route('**/api/media/**', route => route.fulfill({
    contentType: 'image/png', body: previewPng,
  }));
}

async function login(page) {
  await page.getByLabel('用户名').fill('admin');
  await page.getByLabel('密码').fill('secret');
  const response = page.waitForResponse('**/auth/**');
  await page.getByRole('button', { name: '登录', exact: true }).click();
  expect(await (await response).json()).toEqual(admin);
  await expect(page.locator('.login-overlay')).toHaveCount(0);
}

test('Vue shell retains existing workflow labels across desktop and mobile', async ({ page }) => {
  await installRouteMocks(page);
  await page.goto('/');
  await expect(page.getByText('图片溯源系统').first()).toBeVisible();
  await expect(page.getByRole('button', { name: '登录', exact: true })).toBeVisible();
  await page.screenshot({ path: 'test-results/login-desktop.png', fullPage: true });

  await login(page);
  await expect(page.getByText('生成水印').first()).toBeVisible();
  await page.screenshot({ path: 'test-results/watermark-desktop.png', fullPage: true });

  await page.getByRole('button', { name: '图片溯源' }).click();
  await page.getByPlaceholder('https://example.com/photo.jpg 或 /api/media/...').fill('https://example.com/source.jpg');
  await page.getByRole('button', { name: '开始 V4 溯源' }).click();
  await expect(page.getByText('trace-1')).toBeVisible();
  await page.screenshot({ path: 'test-results/trace-desktop.png', fullPage: true });

  await page.getByRole('button', { name: '图片管理' }).click();
  await expect(page.getByText('sample.jpg')).toBeVisible();
  await page.screenshot({ path: 'test-results/management-desktop.png', fullPage: true });
  await page.getByTitle('溯源').click();
  await expect(page.getByPlaceholder('https://example.com/photo.jpg 或 /api/media/...')).toHaveValue('/api/media/record-1');
  const managedPreview = page.locator('.managed-trace-preview img');
  await expect(managedPreview).toBeVisible();
  await expect.poll(() => managedPreview.evaluate(image => image.naturalWidth)).toBeGreaterThan(0);
  await page.screenshot({ path: 'test-results/managed-trace-desktop.png', fullPage: true });

  await page.getByRole('button', { name: '角色管理' }).click();
  await expect(page.getByText('菜单权限')).toBeVisible();
  await page.screenshot({ path: 'test-results/role-desktop.png', fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByText('菜单权限')).toBeVisible();
  await page.getByRole('button', { name: '打开菜单' }).click();
  await expect(page.locator('.nav-links')).toHaveClass(/is-open/);
  await expect(page.getByRole('button', { name: '图片管理' })).toBeVisible();
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/role-mobile.png', fullPage: true });
  await page.getByRole('button', { name: '图片管理' }).click();
  await expect(page.getByText('2.5 MB')).toBeVisible();
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/management-mobile.png', fullPage: true });
});
