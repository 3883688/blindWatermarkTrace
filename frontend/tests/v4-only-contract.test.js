import { expect, test, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import {
  embedWatermark,
  extractUpload,
  extractUrl,
  listImages,
  v4Capabilities,
} from '../src/api/trace.js';
import { createWatermarkForm, watermarkFormData } from '../src/forms/watermark.js';

const ok = () => Promise.resolve(new Response('{}', { status: 200 }));

test('product requests use only V4 endpoints', async () => {
  const fetchMock = vi.fn(ok);
  vi.stubGlobal('fetch', fetchMock);
  const form = new FormData();

  await embedWatermark(form);
  await extractUpload(form);
  await extractUrl(form);
  await listImages();
  await v4Capabilities();

  expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
    '/api/v4/generate', '/api/v4/detect', '/api/v4/detect-url',
    '/api/v4/records', '/api/v4/capabilities',
  ]);
});

test('generation form contains only file and exact V4 codec', () => {
  const file = new File(['image'], 'source.png', { type: 'image/png' });
  const data = watermarkFormData(file, createWatermarkForm());
  expect([...data.entries()].map(([key, value]) => [key, value instanceof File ? value.name : value]))
    .toEqual([['file', 'source.png'], ['codec', 'hmac64_rs_16_8_split_repeat_sync_v4']]);
});

test('V4 views contain no legacy algorithms or real upload paths', () => {
  const source = [
    readFileSync('src/views/WatermarkView.vue', 'utf8'),
    readFileSync('src/views/TraceView.vue', 'utf8'),
  ].join('\n');
  for (const legacy of ['DCT', 'DWT', 'FFT', 'LSB', '/uploads/']) {
    expect(source).not.toContain(legacy);
  }
});
