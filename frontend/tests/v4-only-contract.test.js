import { expect, test, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import {
  embedWatermark,
  extractUpload,
  extractUrl,
  listImages,
} from '../src/api/trace.js';
import { createWatermarkForm, watermarkFormData } from '../src/forms/watermark.js';

const ok = () => Promise.resolve(new Response('{}', { status: 200 }));

test('original product paths use the V4-only compatibility API', async () => {
  const fetchMock = vi.fn(ok);
  vi.stubGlobal('fetch', fetchMock);
  const form = new FormData();

  await embedWatermark(form);
  await extractUpload(form);
  await extractUrl(form);
  await listImages();

  expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
    '/api/watermark/embed', '/api/watermark/extract',
    '/api/watermark/extract-url', '/api/images',
  ]);
});

test('generation form pins the original UI to V4', () => {
  const file = new File(['image'], 'source.png', { type: 'image/png' });
  const data = watermarkFormData(file, createWatermarkForm());
  expect(data.get('file')).toBe(file);
  expect(data.get('robust_watermark_version')).toBe('4');
});

test('V4 views contain no real upload paths', () => {
  const source = [
    readFileSync('src/views/WatermarkView.vue', 'utf8'),
    readFileSync('src/views/TraceView.vue', 'utf8'),
  ].join('\n');
  expect(source).not.toContain('/uploads/');
  expect(source).toContain('/api/media/');
});
