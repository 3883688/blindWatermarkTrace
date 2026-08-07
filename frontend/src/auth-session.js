export const AUTH_INVALID_EVENT = 'trace:authentication-invalid';

export function authHeaders() {
  try {
    const user = JSON.parse(localStorage.getItem('currentUser') || 'null');
    return user?.token ? { Authorization: `Bearer ${user.token}` } : {};
  } catch {
    return {};
  }
}

export function invalidateAuthentication() {
  localStorage.removeItem('currentUser');
  window.dispatchEvent(new Event(AUTH_INVALID_EVENT));
}
