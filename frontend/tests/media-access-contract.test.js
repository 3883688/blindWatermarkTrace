import { readFile } from 'node:fs/promises';
import { expect, test } from 'vitest';

test('browser media elements prefer short-lived signed access urls', async () => {
  const [watermark, table] = await Promise.all([
    readFile('src/views/WatermarkView.vue', 'utf8'),
    readFile('src/components/ImageTable.vue', 'utf8'),
  ]);

  expect(watermark).toContain('safeImageUrl(result.output_access_url)');
  expect(table).toContain('record.thumbnail_access_url || record.output_access_url');
  expect(table).toContain('safeImageUrl(record.output_access_url)');
  expect(table).toContain(":disabled=\"!download(record)\"");
  expect(table).toContain("emit('download', download(record))");
  expect(table).not.toContain('record.download_url');
});
