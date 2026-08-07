import { request } from './client.js';

const json = body => ({
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

const segment = value => encodeURIComponent(value);

export function login(username, password) {
  const body = new FormData();
  body.append('username', username);
  body.append('password', password);
  return request('/auth/login', { method: 'POST', body, authenticated: false });
}

export const embedWatermark = body => request('/api/watermark/embed', { method: 'POST', body });
export const extractUpload = body => request('/api/watermark/extract', { method: 'POST', body });
export const extractUrl = body => request('/api/watermark/extract-url', { method: 'POST', body });

export const listImages = () => request('/api/images');
export const deleteImage = id => request(`/api/images/${segment(id)}`, { method: 'DELETE' });

export const listRoles = () => request('/api/roles');
export const saveRole = (key, menus) => request(`/api/roles/${segment(key)}`, {
  method: 'PUT',
  ...json({ menus }),
});

export const listUsers = () => request('/api/users');
export const createUser = payload => request('/api/users', { method: 'POST', ...json(payload) });
export const saveUser = (username, role) => request(`/api/users/${segment(username)}`, {
  method: 'PUT',
  ...json({ role }),
});

export const dashboardStats = () => request('/api/dashboard-stats');
