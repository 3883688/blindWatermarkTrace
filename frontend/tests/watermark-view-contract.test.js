import { describe, expect, test } from 'vitest';
import { createAsyncGuard, safeImageUrl, syncConfidence } from '../src/views/result-format.js';

describe('watermark and trace result contracts', () => {
  test('does not turn absent V4 synchronization evidence into a zero confidence', () => {
    expect(syncConfidence(undefined)).toBe('-');
    expect(syncConfidence(null)).toBe('-');
    expect(syncConfidence(0.875)).toBe('87.5%');
  });

  test('only accepts existing absolute HTTP or application-relative result image links', () => {
    expect(safeImageUrl('https://example.test/image.png')).toBe('https://example.test/image.png');
    expect(safeImageUrl('/uploads/image.png')).toBe('/uploads/image.png');
    expect(safeImageUrl('//evil.example/image.png')).toBe('');
    expect(safeImageUrl('javascript:alert(1)')).toBe('');
  });

  test('stops asynchronous view continuations after unmount', () => {
    const guard = createAsyncGuard();
    expect(guard.isActive()).toBe(true);
    guard.dispose();
    expect(guard.isActive()).toBe(false);
  });
});
