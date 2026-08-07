import { readFile } from 'node:fs/promises';
import { expect, test } from 'vitest';

test('browser media elements prefer short-lived signed access urls', async () => {
  const [watermark, table] = await Promise.all([
    readFile('src/views/WatermarkView.vue', 'utf8'),
    readFile('src/components/ImageTable.vue', 'utf8'),
  ]);

  expect(watermark).toContain('result.download_access_url || result.download_url');
  expect(watermark).toContain('result.original_access_url || result.original_url');
  expect(table).toContain('record.thumbnail_access_url || record.thumbnail_url');
  expect(table).toContain('record.download_access_url || record.download_url');
  expect(table).toContain(":disabled=\"!download(record)\"");
  expect(table).toContain("emit('download', download(record))");
  expect(table).not.toContain('safeImageUrl(record.download_url)');
});
