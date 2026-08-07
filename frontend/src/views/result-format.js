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

export function traceResultFromRecord(record) {
  if (!record || record.robust_watermark_version !== 4) return null;
  return {
    user_id: record.user_id,
    trace_id: record.trace_id,
    evidence_uuid: record.evidence_uuid,
    evidence_uuid_head: record.evidence_uuid_head,
    evidence_uuid_tail: record.evidence_uuid_tail,
    robust_watermark_version: record.robust_watermark_version,
    mode_label: record.mode_label,
    created_at: record.created_at,
    confidence: record.confidence || 98,
    code_recovery: record.code_recovery,
    status: record.status,
    extracted_at: new Date().toLocaleString(),
  };
}
