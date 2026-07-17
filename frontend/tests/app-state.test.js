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

  test('adds role management only for the configured administrator and persists normalized themes', () => {
    const state = createAppState({ adminUser: 'system-owner' });

    state.setUser({ username: 'system-owner', role: 'admin', menus: ['watermark'] });
    expect(state.visibleMenus).toEqual(['watermark', 'role']);

    state.setTheme('unexpected');
    expect(state.theme).toBe('dark');
    expect(localStorage.getItem('siteTheme')).toBe('dark');
  });

  test('does not grant role management from a forged admin role', () => {
    const state = createAppState({ adminUser: 'system-owner' });
    state.setUser({ username: 'untrusted-user', role: 'admin', menus: ['watermark', 'role'] });

    expect(state.visibleMenus).toEqual(['watermark']);
    state.selectPage('role');
    expect(state.activePage).toBe('watermark');
  });
});
