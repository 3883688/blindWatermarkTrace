import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { expect, test } from 'vitest';

test('native dialog cancellation and close events synchronize the parent open state', async () => {
  const component = await readFile(resolve(process.cwd(), 'src/components/ResultDialog.vue'), 'utf8');
  expect(component).toContain('@cancel.prevent="emit(\'close\')"');
  expect(component).toContain('@close="emit(\'close\')"');
});
