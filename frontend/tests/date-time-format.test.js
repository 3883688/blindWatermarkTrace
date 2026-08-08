import { expect, test } from 'vitest';

import { formatDateTime } from '../src/views/result-format.js';

test('formats record timestamps as year-month-day hour:minute:second', () => {
  expect(formatDateTime('2026-08-08T09:35:37.482+08:00')).toBe('2026-08-08 09:35:37');
  expect(formatDateTime('2026-08-08 09:35:37')).toBe('2026-08-08 09:35:37');
  expect(formatDateTime('')).toBe('-');
});
