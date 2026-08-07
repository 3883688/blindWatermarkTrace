function csvCell(value) {
  return `"${String(value ?? '').replace(/"/g, '""')}"`;
}

export function setPageSelection(selected, ids, checked) {
  const next = new Set(selected);
  ids.filter(Boolean).forEach(id => checked ? next.add(id) : next.delete(id));
  return next;
}

export function reconcileSelectedIds(selected, records) {
  const ids = new Set(records.map(record => record.id));
  return new Set([...selected].filter(id => ids.has(id)));
}

export function csvRows(records) {
  const header = ['文件名', '用户', 'Trace ID', '水印版本', '嵌入时间', '状态', '置信度', '原图', '水印图'];
  return [header.map(csvCell).join(','), ...records.map(record => [
    record.name, record.user_id, record.trace_id, 'V4',
    record.created_at || record.time, record.status, Number(record.confidence || record.conf || 0),
    record.original_url, record.download_url,
  ].map(csvCell).join(','))];
}

export function downloadCsv(records) {
  const blob = new Blob([`\uFEFF${csvRows(records).join('\n')}`], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `watermark-images-${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
