import { reactive } from 'vue';

const USER_KEY = 'currentUser';
const THEME_KEY = 'siteTheme';
const DEFAULT_MENUS = ['watermark', 'trace', 'manage'];
const CONFIGURED_ADMIN_USER = String(import.meta.env.VITE_ADMIN_USER || '').trim();

function readUser() {
  try {
    const saved = localStorage.getItem(USER_KEY);
    return saved ? JSON.parse(saved) : null;
  } catch {
    return null;
  }
}

function readTheme() {
  return localStorage.getItem(THEME_KEY) === 'light' ? 'light' : 'dark';
}

export function createAppState({ adminUser = CONFIGURED_ADMIN_USER } = {}) {
  const isConfiguredAdmin = user => Boolean(adminUser) && user?.username === adminUser;
  return reactive({
    currentUser: readUser(),
    theme: readTheme(),
    activePage: 'watermark',
    get visibleMenus() {
      const menus = this.currentUser?.menus || DEFAULT_MENUS;
      if (isConfiguredAdmin(this.currentUser)) {
        return [...new Set([...menus, 'role'])];
      }
      return menus.filter(menu => menu !== 'role');
    },
    setUser(user) {
      this.currentUser = user;
      localStorage.setItem(USER_KEY, JSON.stringify(user));
      this.ensureActivePage();
    },
    clearUser() {
      this.currentUser = null;
      localStorage.removeItem(USER_KEY);
      this.ensureActivePage();
    },
    setTheme(theme) {
      this.theme = theme === 'light' ? 'light' : 'dark';
      localStorage.setItem(THEME_KEY, this.theme);
    },
    selectPage(page) {
      if (this.visibleMenus.includes(page)) this.activePage = page;
    },
    ensureActivePage() {
      if (!this.visibleMenus.includes(this.activePage)) {
        this.activePage = this.visibleMenus[0] || 'trace';
      }
    },
  });
}
