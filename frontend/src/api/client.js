import { authHeaders, invalidateAuthentication } from '../auth-session.js';

const fallbackMessage = status => `请求失败 (${status})`;

async function parseJson(response) {
  try {
    return await response.json();
  } catch {
    return undefined;
  }
}

/**
 * Calls an existing FastAPI route and exposes one consistent error shape to
 * Vue views. API modules retain responsibility for the request contract.
 */
export async function request(path, options = {}) {
  const { authenticated = true, ...fetchOptions } = options;
  const headers = {
    ...(fetchOptions.headers || {}),
    ...(authenticated ? authHeaders() : {}),
  };
  let response;
  try {
    response = await fetch(path, { ...fetchOptions, headers });
  } catch (error) {
    throw new Error(error?.message || '网络请求失败');
  }

  const body = await parseJson(response);
  if (!response.ok) {
    if (authenticated && response.status === 401) invalidateAuthentication();
    throw new Error(body?.detail || body?.message || fallbackMessage(response.status));
  }
  if (body === undefined) {
    throw new Error('响应格式无效');
  }
  return body;
}
