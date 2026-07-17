import { reactive } from 'vue';

const USER_KEY = 'currentUser';
const THEME_KEY = 'siteTheme';

function readUser() {
  try {
    const saved = localStorage.getItem(USER_KEY);
    return saved ? JSON.parse(saved) : null;
  } catch {
    return null;
  }
}

function readTheme() {
  return localStorage.getItem(THEME_KEY) || 'dark';
}

export function createAppState() {
  return reactive({
    currentUser: readUser(),
    theme: readTheme(),
    activePage: 'watermark',
    get visibleMenus() {
      return this.currentUser?.menus || [];
    },
    setUser(user) {
      this.currentUser = user;
      localStorage.setItem(USER_KEY, JSON.stringify(user));
    },
    clearUser() {
      this.currentUser = null;
      localStorage.removeItem(USER_KEY);
    },
    setTheme(theme) {
      this.theme = theme;
      localStorage.setItem(THEME_KEY, theme);
    },
  });
}
