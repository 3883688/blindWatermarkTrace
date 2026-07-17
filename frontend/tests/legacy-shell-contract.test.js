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

test('shell controls retain the original selector contracts and dialog backdrops', async () => {
  const css = await readFile('src/styles/index.css', 'utf8');

  expect(css).toContain('.btn-outline{display:inline-flex;align-items:center;gap:8px;padding:10px 20px;background:transparent;border:1px solid rgba(255,255,255,0.15);border-radius:8px;font-size:13px;color:rgba(255,255,255,0.65);cursor:pointer;transition:all .2s}');
  expect(css).toContain('.icon-btn{width:34px;height:34px;border-radius:7px;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);color:#e2e8f0;cursor:pointer;font-size:17px;transition:all .2s}');
  expect(css).toContain('.or-divider:before,.or-divider:after{content:\'\';flex:1;height:1px;background:rgba(255,255,255,0.08)}');
  expect(css).toContain('.page-btns{display:flex;gap:4px}');
  expect(css).toContain('.advanced-section-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:14px}');
  expect(css).toContain('.result-dialog::backdrop{background:rgba(0,0,0,0.62);backdrop-filter:blur(3px)}');
  expect(css).toContain('.image-preview-dialog::backdrop{background:rgba(0,0,0,0.68);backdrop-filter:blur(4px)}');
});
