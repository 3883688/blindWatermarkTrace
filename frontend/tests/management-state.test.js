import { expect, test } from 'vitest';
import {
  csvRows,
  reconcileSelectedIds,
  setPageSelection,
} from '../src/state/management.js';

test('page selection only changes IDs visible on the current page', () => {
  const selected = new Set(['persisted']);

  expect([...setPageSelection(selected, ['one', 'two'], true)]).toEqual(['persisted', 'one', 'two']);
  expect([...setPageSelection(selected, ['one', 'two'], false)]).toEqual(['persisted']);
});

test('image reload keeps selected records that still exist', () => {
  expect([...reconcileSelectedIds(new Set(['one', 'gone']), [{ id: 'one' }, { id: 'two' }])]).toEqual(['one']);
});

test('CSV rows export the V4-only watermark version', () => {
  expect(csvRows([{ name: 'a"b.png', user_id: 'alice', trace_id: 't1', mode: 'dct', created_at: '2026-07-17', status: '保护中', confidence: 98, original_url: '/original', download_url: '/download' }])).toEqual([
    '"文件名","用户","Trace ID","水印版本","嵌入时间","状态","置信度","原图","水印图"',
    '"a""b.png","alice","t1","V4","2026-07-17","保护中","98","/original","/download"',
  ]);
});
