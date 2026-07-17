import { expect, test } from 'vitest';
import { createImageState } from '../src/state/images.js';

const rows = [
  {
    id: 'new', name: 'contract.png', user_id: 'alice', trace_id: 'trace-a',
    mode: 'hybrid', mode_label: '全部算法', status: '保护中',
    created_at: '2026-07-17T10:00:00', confidence: 25, hidden_note: 'needle',
  },
  {
    id: 'old', name: 'portrait.png', user_id: 'bob', trace_id: 'trace-b',
    mode: 'dct', mode_label: 'DCT + 空间域', status: '溯源命中',
    created_at: '2026-07-16T10:00:00', conf: 90,
  },
  {
    id: 'mid', name: 'landscape.png', user_id: 'carol', trace_id: 'trace-c',
    mode: 'lsb', mode_label: '仅空间域', status: '泄露预警',
    created_at: '2026-07-17T08:00:00', confidence: 60,
  },
];

test('image state matches the original search, status, mode, and date filters', () => {
  const state = createImageState(rows, 10);

  state.search = 'trace-b';
  expect(state.filteredRows.value.map(row => row.id)).toEqual(['old']);

  state.search = 'needle';
  expect(state.filteredRows.value).toEqual([]);

  state.search = '';
  state.activeFilter = '泄露预警';
  expect(state.filteredRows.value.map(row => row.id)).toEqual(['mid']);

  state.activeFilter = '全部';
  state.mode = 'DCT + 空间域';
  expect(state.filteredRows.value.map(row => row.id)).toEqual(['old']);

  state.mode = '';
  state.date = '2026-07-17';
  expect(state.filteredRows.value.map(row => row.id)).toEqual(['new', 'mid']);
});

test('image state searches each original searchable field and excludes unrelated fields', () => {
  const searchableFields = [
    'name',
    'user_id',
    'trace_id',
    'mode_label',
    'mode',
    'status',
  ];

  for (const field of searchableFields) {
    const state = createImageState([{ id: field, [field]: `match-${field}` }]);
    state.search = `MATCH-${field}`;
    expect(state.filteredRows.value.map(row => row.id)).toEqual([field]);
  }

  const state = createImageState([{ id: 'hidden', name: 'visible', hidden_note: 'not-searchable' }]);
  state.search = 'not-searchable';
  expect(state.filteredRows.value).toEqual([]);
});

test('image state preserves original created and confidence sort options', () => {
  const state = createImageState(rows, 10);

  expect(state.filteredRows.value.map(row => row.id)).toEqual(['new', 'mid', 'old']);
  state.sort = 'created_asc';
  expect(state.filteredRows.value.map(row => row.id)).toEqual(['old', 'mid', 'new']);
  state.sort = 'confidence_desc';
  expect(state.filteredRows.value.map(row => row.id)).toEqual(['old', 'mid', 'new']);
  state.sort = 'confidence_asc';
  expect(state.filteredRows.value.map(row => row.id)).toEqual(['new', 'mid', 'old']);
});

test('all filter and sort changes reset pagination and paged rows clamp the state page', () => {
  const state = createImageState(rows, 1);

  state.currentPage = 3;
  state.activeFilter = '保护中';
  expect(state.currentPage).toBe(1);
  state.activeFilter = '全部';
  state.currentPage = 3;
  state.search = 'trace-a';
  expect(state.currentPage).toBe(1);
  state.search = '';
  state.currentPage = 3;
  state.mode = 'dct';
  expect(state.currentPage).toBe(1);
  state.mode = '';
  state.currentPage = 3;
  state.date = '2026-07-17';
  expect(state.currentPage).toBe(1);
  state.currentPage = 3;
  state.sort = 'created_asc';
  expect(state.currentPage).toBe(1);
  state.currentPage = 99;
  expect(state.pagedRows.value.map(row => row.id)).toEqual(['new']);
  expect(state.currentPage).toBe(2);
});
