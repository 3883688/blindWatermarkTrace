import { afterEach, describe, expect, test } from 'vitest';
import { createAppState } from '../src/state/app.js';

afterEach(() => localStorage.clear());

describe('application shell state', () => {
  test('keeps the legacy default menus and returns to the first permitted page', () => {
    const state = createAppState();

    expect(state.visibleMenus).toEqual(['watermark', 'trace', 'manage']);
    state.selectPage('manage');
    expect(state.activePage).toBe('manage');

    state.setUser({ username: 'operator', role: 'operator', menus: ['trace'] });
    expect(state.visibleMenus).toEqual(['trace']);
    expect(state.activePage).toBe('trace');
    state.selectPage('manage');
    expect(state.activePage).toBe('trace');
  });

  test('adds role management for the admin role returned by the existing login API and persists normalized themes', () => {
    const state = createAppState();

    state.setUser({ username: 'system-owner', role: 'admin', menus: ['watermark'] });
    expect(state.visibleMenus).toEqual(['watermark', 'role']);

    state.setTheme('unexpected');
    expect(state.theme).toBe('dark');
    expect(localStorage.getItem('siteTheme')).toBe('dark');
  });

  test('does not grant role management to a non-admin login response', () => {
    const state = createAppState();
    state.setUser({ username: 'operator', role: 'operator', menus: ['watermark', 'role'] });

    expect(state.visibleMenus).toEqual(['watermark']);
    state.selectPage('role');
    expect(state.activePage).toBe('watermark');
  });
});
