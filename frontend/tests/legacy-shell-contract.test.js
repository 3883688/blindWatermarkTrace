import { readFile } from 'node:fs/promises';
import { afterEach, expect, test } from 'vitest';
import { createAppState } from '../src/state/app.js';

afterEach(() => localStorage.clear());

test('role management uses the legacy strict admin gate', () => {
  const state = createAppState();

  state.setUser({ username: 'admin', role: 'administrator', menus: ['watermark', 'role'] });
  expect(state.visibleMenus).toEqual(['watermark']);

  state.setUser({ username: 'admin', role: 'admin', menus: ['watermark'] });
  expect(state.visibleMenus).toEqual(['watermark', 'role']);
});

test('website branding uses the configured bilingual title', async () => {
  const [html, navigation, login] = await Promise.all([
    readFile('../index.html', 'utf8'),
    readFile('src/components/AppNavigation.vue', 'utf8'),
    readFile('src/components/LoginOverlay.vue', 'utf8'),
  ]);

  const title = '图片溯源系统（Watermark System）';
  expect(html).toContain(`<title>${title}</title>`);
  expect(navigation).toContain(title);
  expect(login).toContain(title);
});

test('shell controls retain the original selector contracts and dialog backdrops', async () => {
  const [css, index, navigation, trace, user] = await Promise.all([
    readFile('src/styles/legacy.css', 'utf8'),
    readFile('src/styles/index.css', 'utf8'),
    readFile('src/components/AppNavigation.vue', 'utf8'),
    readFile('src/views/TraceView.vue', 'utf8'),
    readFile('src/views/UserView.vue', 'utf8'),
  ]);

  expect(css).toContain('.btn-outline{display:inline-flex;align-items:center;gap:8px;padding:10px 20px;background:transparent;border:1px solid rgba(255,255,255,0.15);border-radius:8px;font-size:13px;color:rgba(255,255,255,0.65);cursor:pointer;transition:all .2s}');
  expect(css).toContain('.icon-btn{width:34px;height:34px;border-radius:7px;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);color:#e2e8f0;cursor:pointer;font-size:17px;transition:all .2s}');
  expect(css).toContain('.or-divider:before,.or-divider:after{content:\'\';flex:1;height:1px;background:rgba(255,255,255,0.08)}');
  expect(css).toContain('.page-btns{display:flex;gap:4px}');
  expect(css).toContain('.advanced-section-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:14px}');
  expect(css).toContain('.result-dialog::backdrop{background:rgba(0,0,0,0.62);backdrop-filter:blur(3px)}');
  expect(css).toContain('.image-preview-dialog::backdrop{background:rgba(0,0,0,0.68);backdrop-filter:blur(4px)}');
  expect(navigation).toContain('<div class="user-avatar">BX</div>');
  expect(index).toContain('.dropzone.compact{padding:32px}');
  expect(index).toContain('.trace-button{margin-top:20px;width:100%;justify-content:center}');
  expect(index).toContain('.url-btn{line-height:1}');
  expect(index).toContain('.trace-result-card>.result-title i{font-size:18px}');
  expect(index).toContain('.result-val.owner{color:#818cf8}');
  expect(index).toContain('.trace-help{font-size:13px;color:rgba(255,255,255,0.45);line-height:1.8}');
  expect(index).toContain('[data-theme="light"] .trace-help{color:#64748b}');
  expect(index).toContain('[data-theme="light"] .badge-green{background:rgba(22,163,74,0.10);color:#15803d;border-color:rgba(21,128,61,0.25)}');
  expect(index).toContain('.s-icon.indigo{background:rgba(99,102,241,0.15);color:#818cf8}');
  expect(index).toContain('.img-table td.small-cell{font-size:12px}');
  expect(index).toContain('.img-table td.date-cell{color:rgba(255,255,255,0.45);font-size:12px}');
  expect(index).toContain('.confidence{display:flex;align-items:center;gap:8px}');
  expect(index).toContain('.confidence .confidence-bar{width:60px}');
  expect(index).toContain('.page-info strong{color:#e2e8f0}');
  expect(index).toContain('.img-table input[type="checkbox"]{position:relative;top:1.5px}');
  expect(index).toContain('.img-table .status-badge{position:relative;top:-1px}');
  expect(index).toContain('.user-actions{display:flex;gap:8px;align-items:center}');
  expect(trace).toContain('<span class="result-time">-</span>');
  expect([...user.matchAll(/class="field-group" style="margin-bottom:0"/g)]).toHaveLength(3);
});
