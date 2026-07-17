export function syncConfidence(value) {
  if (value === undefined || value === null || value === '') return '-';
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(1)}%` : '-';
}

export function safeImageUrl(value) {
  const url = String(value || '');
  return /^(https?:\/\/|\/(?!\/))/i.test(url) ? url : '';
}

export function createAsyncGuard() {
  let active = true;
  return { isActive: () => active, dispose: () => { active = false; } };
}
